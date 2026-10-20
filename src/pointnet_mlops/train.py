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

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    "train_acc",
    "val_loss",
    "val_acc",
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


def save_atomic(obj, path):
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

    start_epoch, best_val_acc = 0, -1.0
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
        best_val_acc = checkpoint["best_val_acc"]
        print(
            f"resumed from {CHECKPOINT_PATH} at epoch {start_epoch} (best val acc {best_val_acc:.4f})"
        )
    reset_log(keep_before_epoch=start_epoch)

    for epoch in range(start_epoch, train_params["epochs"]):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            classifier,
            train_loader,
            device,
            epoch,
            optimizer,
            train_params["feature_transform_weight"],
        )
        val_loss, val_acc = run_epoch(classifier, val_loader, device, epoch)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "seconds": round(time.time() - start, 1),
        }
        append_log(row)
        print(
            f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:g} | {row['seconds']}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_atomic(
                {
                    "model": classifier.state_dict(),
                    "feature_transform": feature_transform,
                    "classes": CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                MODEL_PATH,
            )
            print(f"  new best val acc, saved {MODEL_PATH}")

        save_atomic(
            {
                "model": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": get_rng_state(),
                "loader_generator": loader_generator.get_state(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            },
            CHECKPOINT_PATH,
        )

    print(f"done: best val acc {best_val_acc:.4f}, model at {MODEL_PATH}")


if __name__ == "__main__":
    main()
