"""Train PointNet on the prepared ModelNet10 train split, selecting the best epoch on val.

Outputs (fixed paths):
  models/model.pt          weights from the epoch with the best val accuracy
  checkpoints/last.pt      full state after the latest epoch, for --resume
  reports/train_log.csv    per-epoch loss, accuracy and learning rate
"""

import argparse
import csv
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
import mlflow.pytorch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from pointnet_mlops.data import CLASSES, NUM_CLASSES, ModelNet10Dataset
from pointnet_mlops.model import PointNetCls, feature_transform_regularizer
from pointnet_mlops.utils import (
    get_device,
    get_rng_state,
    load_params,
    seed_everything,
    set_rng_state,
)

MODEL_PATH = "models/model.pt"
CHECKPOINT_PATH = "checkpoints/last.pt"
LOG_PATH = "reports/train_log.csv"
LOG_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "seconds",
]


def run_epoch(
    classifier, loader, device, epoch, optimizer=None, feature_transform_weight=0.0
):
    """One pass over loader: trains if an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy). Raises FloatingPointError on a non-finite loss.
    """
    training = optimizer is not None
    split = "train" if training else "val"
    classifier.train(training)
    total_loss, total_correct, total_seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for i, (points, target) in enumerate(
            tqdm(loader, desc=f"epoch {epoch} {split}", leave=False)
        ):
            points, target = points.transpose(2, 1).to(device), target.to(device)
            pred, _, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if training and trans_feat is not None:
                loss = loss + feature_transform_weight * feature_transform_regularizer(
                    trans_feat
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{split} loss is {loss.item()} at epoch {epoch}, batch {i}"
                )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = len(target)
            total_loss += loss.item() * n
            total_correct += (pred.argmax(dim=1) == target).sum().item()
            total_seen += n

    return total_loss / total_seen, total_correct / total_seen


def save_model(obj, path):
    # Write to a temp file first so a crash mid-save never leaves a truncated file behind
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def reset_log(keep_before_epoch):
    """Rewrite the log keeping only rows before keep_before_epoch (none on a fresh run)."""
    rows = []
    if keep_before_epoch > 0 and os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if int(row["epoch"]) < keep_before_epoch
            ]
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_log(row):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--resume", action="store_true", help=f"continue from {CHECKPOINT_PATH}"
    )
    args = parser.parse_args()

    mlflow.set_experiment("PointNet")
    # mlflow.set_tracking_uri("localhost:8000")

    params = load_params(args.params)
    data_params, model_params, train_params = (
        params["data"],
        params["model"],
        params["train"],
    )
    # PyYAML reads exponents without a decimal point (e.g. 1e-4) as strings
    for key in ("lr", "lr_gamma", "feature_transform_weight"):
        train_params[key] = float(train_params[key])

    seed_everything(train_params["seed"])
    device = get_device()
    print(f"device: {device}")

    # Drives shuffling and DataLoader worker seeds, so it is saved and restored with the checkpoint
    loader_generator = torch.Generator().manual_seed(train_params["seed"])
    train_loader = DataLoader(
        ModelNet10Dataset(data_params["prepared_dir"], "train", data_augmentation=True),
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=train_params["num_workers"],
        generator=loader_generator,
    )
    val_loader = DataLoader(
        ModelNet10Dataset(data_params["prepared_dir"], "val", data_augmentation=False),
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=train_params["num_workers"],
    )
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    feature_transform = model_params["feature_transform"]
    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=feature_transform).to(
        device
    )
    optimizer = optim.Adam(
        classifier.parameters(), lr=train_params["lr"], betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_params["lr_step_size"],
        gamma=train_params["lr_gamma"],
    )

    start_epoch, best_val_accuracy = 0, -1.0
    if args.resume:
        # weights_only=False: the checkpoint holds Python/NumPy RNG state, not just tensors.
        # map_location="cpu": RNG states must stay on CPU; load_state_dict moves the rest to device.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        classifier.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        set_rng_state(checkpoint["rng"])
        loader_generator.set_state(checkpoint["loader_generator"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["best_val_accuracy"]
        print(
            f"resumed from {CHECKPOINT_PATH} at epoch {start_epoch} (best val acc {best_val_accuracy:.4f})"
        )
    reset_log(keep_before_epoch=start_epoch)

    with mlflow.start_run() as run:
        # log hyperparameters
        mlflow.log_params(params=train_params)

        for epoch in range(start_epoch, train_params["epochs"]):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = run_epoch(
                classifier,
                train_loader,
                device,
                epoch,
                optimizer,
                train_params["feature_transform_weight"],
            )
            val_loss, val_accuracy = run_epoch(classifier, val_loader, device, epoch)
            
            # Log metrics at each training epoch
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_accuracy}, step=epoch)
            mlflow.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy}, step=epoch)
            mlflow.log_metrics({"learning_rate": lr}, step=epoch)
            
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "seconds": round(time.time() - start, 1),
            }
            append_log(row)
            print(
                f"epoch {epoch}: train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | lr {lr:g} | {row['seconds']}s"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_model(
                    {
                        "model": classifier.state_dict(),
                        "feature_transform": feature_transform,
                        "classes": CLASSES,
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                    },
                    MODEL_PATH,
                )
                print(f"  new best val acc, saved {MODEL_PATH}")
                
                
            save_model(
                {
                    "model": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": get_rng_state(),
                    "loader_generator": loader_generator.get_state(),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
                CHECKPOINT_PATH,
            )

        # set input example to log model to mlflow
        num_points = data_params["num_points"]
        input_example = np.random.rand(2, 3, num_points).astype(np.float32)
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 3, num_points))]),  # -1 = any batch size
            outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, NUM_CLASSES))]),
        )


        # log final best model in the training run
        best_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        classifier.load_state_dict(best_model["model"])
        classifier.to("cpu").eval()

        mlflow.log_metrics({"best_val_accuracy": best_model["val_accuracy"], "best_epoch": best_model["epoch"]})
        mlflow.pytorch.log_model(
            pytorch_model=classifier, 
            name="pointnet-model", 
            input_example=input_example, 
            signature=signature
        )
       

    print(f"done: best val acc {best_val_accuracy:.4f}, model at {MODEL_PATH}")


if __name__ == "__main__":
    main()
