"""Train PointNet on the prepared ModelNet10 data train split, selecting the best epoch on val.

With two or more CUDA devices visible, training is data-parallel: one process per GPU,
each wrapping its own model replica in DistributedDataParallel and training on a
different shard of every epoch. Otherwise it runs as a single process on one device.

Outputs (fixed paths):
  models/model.pt          weights from the epoch with the best val accuracy
  checkpoints/last.pt      full state after the latest epoch, for --resume
  reports/train_log.csv    per-epoch loss, accuracy and learning rate
"""

import argparse
import contextlib
import csv
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP

import mlflow
import mlflow.pytorch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from mlflow.tracking import MlflowClient

from pointnet_mlops.data import CLASSES, NUM_CLASSES, ModelNet10Dataset
from pointnet_mlops.model import LogProbsOnly, PointNetCls, feature_transform_regularizer
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



def setup_process_group(local_rank, world_size):
    '''
    local_rank: identifier for pariticular GPU on one node
    world: total number of process in a the group
    '''
    os.environ['MASTER_ADDR'] = 'localhost'           # IP address of rank 0 process
    os.environ['MASTER_PORT'] = '12355'               # a free port used to communicate amongst processors

    torch.cuda.set_device(local_rank)
    print(f"Setting up rank {local_rank}")

    # Pick the communication backend that matches the hardware (nccl for CUDA GPUs)
    accelerator = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(accelerator)
    print(f"Accelerator: '{accelerator}'")

    dist.init_process_group(
        backend,                  # backend being used; nccl typically used with distributed GPU training
        rank=local_rank,          # rank of the current process being used
        world_size=world_size     # total number of processors being used
    )

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


def run_epoch(
    classifier,
    loader,
    device,
    epoch,
    optimizer=None,
    feature_transform_weight=0.0,
    show_progress=True,
):
    """One pass over loader: trains if an optimizer is given, otherwise evaluates.

    Returns the sums (loss, correct, seen) over this process's batches, so that they can
    be added up across processes before dividing. Raises FloatingPointError on a
    non-finite loss.
    """
    training = optimizer is not None
    split = "train" if training else "val"

    # Switches dropout and batch norm between their training and evaluation behaviour
    classifier.train(training)
    total_loss, total_correct, total_seen = 0.0, 0, 0

    # Gradients are only tracked when training; evaluation skips them to save time and memory
    with torch.set_grad_enabled(training):
        for i, (points, target) in enumerate(
            tqdm(
                loader,
                desc=f"epoch {epoch} {split}",
                leave=False,
                disable=not show_progress,
            )
        ):
            # move each batch of data to device before going into model
            # The loader yields (batch, points, 3); the model's Conv1d layers want (batch, 3, points)
            points, target = points.transpose(2, 1).to(device), target.to(device)
            
            # pred holds log-probabilities per class; trans_feat is the 64x64 feature
            # transform matrix, or None when the feature transform is switched off
            pred, _, trans_feat = classifier(points)

            # nll_loss on log-probabilities is the usual cross-entropy loss
            loss = F.nll_loss(pred, target)
            if training and trans_feat is not None:
                # Penalty that keeps the feature transform close to a rotation (orthogonal matrix)
                loss = loss + feature_transform_weight * feature_transform_regularizer(trans_feat)

            # Stop at once on NaN/inf rather than carry on training a broken model
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{split} loss is {loss.item()} at epoch {epoch}, batch {i}"
                )

            if training:
                optimizer.zero_grad()
                # With a DDP model, backward() also averages the gradients across all
                # processes, so every replica takes the same optimizer step
                loss.backward()
                optimizer.step()

            # loss is the batch mean, so weight it by batch size to get a sum
            # (the last batch can be smaller than the rest)
            n = len(target)
            total_loss += loss.item() * n
            total_correct += (pred.argmax(dim=1) == target).sum().item()
            total_seen += n

    return total_loss, total_correct, total_seen


def train(local_rank, world_size, args):
    """The training job of one process. Rank 0 also validates, logs and saves."""
    # world_size is the number of training processes (one per GPU); local_rank is this
    # process's index among them, which is also the index of the GPU it uses
    distributed = world_size > 1

    # Every process trains, but only rank 0 validates, logs and writes files, so that
    # nothing is logged or saved more than once
    is_main = local_rank == 0

    if is_main:
        mlflow.set_experiment("PointNet")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

    params = load_params(args.params)
    data_params, model_params, train_params = (
        params["data"],
        params["model"],
        params["train"],
    )
    # PyYAML reads exponents without a decimal point (e.g. 1e-4) as strings
    for key in ("lr", "lr_gamma", "feature_transform_weight"):
        train_params[key] = float(train_params[key])

    # --epochs on the command line takes precedence over params.yaml
    if args.epochs is not None:
        train_params["epochs"] = args.epochs

    seed = train_params["seed"]

    # Offset by rank so each process draws different dropout masks and augmentation noise
    seed_everything(seed + local_rank)

    # Distributed: this process's own GPU. Otherwise: whatever is available (CUDA, MPS or CPU)
    device = torch.device(f"cuda:{local_rank}") if distributed else get_device()
    if is_main:
        print(f"device: {device}  processes: {world_size}")

    # load torch dataset
    train_dataset = ModelNet10Dataset(
        data_params["prepared_dir"], "train", data_augmentation=True
    )

    # Splits every epoch into one disjoint shard per process. Its seed must be the same on
    # every rank: each rank builds the same permutation and takes its own slice of it.
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=seed)
        if distributed
        else None
    )

    # Drives DataLoader worker seeds (and shuffling when there is no sampler), so it is
    # saved and restored with the checkpoint
    loader_generator = torch.Generator().manual_seed(seed + local_rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],  # per process
        shuffle=train_sampler is None,  # the sampler shuffles when there is one
        sampler=train_sampler,
        num_workers=train_params["num_workers"],
        generator=loader_generator,
    )

    # Only rank 0 validates, over the whole val set
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            ModelNet10Dataset(
                data_params["prepared_dir"], "val", data_augmentation=False
            ),
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=train_params["num_workers"],
        )
        print(f"train: {len(train_dataset)}  val: {len(val_loader.dataset)}")

    feature_transform = model_params["feature_transform"]

    # move model weights to device (recursive)
    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=feature_transform).to(device)

    # Each process has its own optimizer. The replicas stay identical because they start
    # from the same weights and apply the same averaged gradients at every step.
    optimizer = optim.Adam(
        classifier.parameters(), lr=train_params["lr"], betas=(0.9, 0.999)
    )
    
    # Multiplies the learning rate by lr_gamma every lr_step_size epochs
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
        
        # Every process loads the same file, so all replicas resume from identical state
        classifier.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["best_val_accuracy"]
        if is_main:
            set_rng_state(checkpoint["rng"])
            loader_generator.set_state(checkpoint["loader_generator"])
            print(
                f"resumed from {CHECKPOINT_PATH} at epoch {start_epoch} (best val acc {best_val_accuracy:.4f})"
            )
        else:
            # The saved RNG states are rank 0's; other ranks start from a fresh seed
            seed_everything(seed + local_rank + start_epoch)
            loader_generator.manual_seed(seed + local_rank + start_epoch)
    
    if is_main:
        # Start the CSV log afresh, or on resume keep only the epochs before start_epoch
        reset_log(keep_before_epoch=start_epoch)

    # Wrap after loading so checkpoints keep plain parameter names (no "module." prefix).
    # `classifier` stays the unwrapped model, used for validation and saving.
    # The wrapper shares its weights with `classifier`: training one updates the other.
    train_model = DDP(classifier, device_ids=[local_rank]) if distributed else classifier

    # Only rank 0 talks to MLflow; nullcontext() is a do-nothing stand-in for the other ranks
    with mlflow.start_run() if is_main else contextlib.nullcontext():
        if is_main:
            # log hyperparameters
            mlflow.log_params(params=train_params)

        for epoch in range(start_epoch, train_params["epochs"]):
            if train_sampler is not None:
                # Without this the sampler reuses the same shuffle every epoch
                train_sampler.set_epoch(epoch)

            start = time.time()
            
            # Read the learning rate now: scheduler.step() below changes it for the next epoch
            lr = optimizer.param_groups[0]["lr"]
            
            # Trains on this process's shard only; returns sums for that shard
            train_totals = run_epoch(
                train_model,
                train_loader,
                device,
                epoch,
                optimizer,
                train_params["feature_transform_weight"],
                show_progress=is_main,
            )
            
            if distributed:
                # Add up loss, correct and seen over all processes. all_reduce replaces the
                # tensor on every rank with the sum of that tensor across all ranks, so the
                # train metrics describe the whole epoch and not just this shard.
                train_totals = torch.tensor(
                    train_totals, dtype=torch.float64, device=device
                )
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
                train_totals = train_totals.tolist()
            train_loss = train_totals[0] / train_totals[2]
            train_accuracy = train_totals[1] / train_totals[2]

            # Every rank steps its own scheduler so learning rates stay equal
            scheduler.step()

            if is_main:
                # No optimizer, so this evaluates. It uses the unwrapped model: validation
                # runs on rank 0 alone and needs no gradient synchronisation.
                val_totals = run_epoch(classifier, val_loader, device, epoch)
                val_loss = val_totals[0] / val_totals[2]
                val_accuracy = val_totals[1] / val_totals[2]

                # Log metrics at each training epoch
                mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_accuracy}, step=epoch)
                mlflow.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy}, step=epoch)
                mlflow.log_metrics({"learning_rate": lr}, step=epoch)

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

                # Model selection: keep the weights of the epoch that scores best on val.
                # This file is what evaluate.py loads.
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

                # Written every epoch (best or not) with everything --resume needs to
                # carry on from here
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

            if distributed:
                # Hold the other ranks until rank 0 has validated and saved
                dist.barrier()

        # Training is over: rank 0 registers the best model (not the last one) with MLflow
        if is_main:
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

            # move model back to CPU before being logged
            classifier.to("cpu").eval()

            mlflow.log_metrics({"best_val_accuracy": best_model["val_accuracy"], "best_epoch": best_model["epoch"]})

            # The exported (pt2) model is locked to CPU. Without this, MLflow reloads it on CUDA to check the input example and warns that it can't.
            os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")

            # Log only the log-probabilities output, which is what the signature declares
            model_info = mlflow.pytorch.log_model(
                pytorch_model=LogProbsOnly(classifier).eval(),
                registered_model_name="pointnet",
                name="pointnet-model",
                input_example=input_example,
                signature=signature
            )
            print(f"Registered version: {model_info.registered_model_version}")

    if is_main:
        print(f"done: best val acc {best_val_accuracy:.4f}, model at {MODEL_PATH}")


def train_worker(local_rank, world_size, args):
    """Entry point of each process: join the process group, train, leave the group."""
    distributed = world_size > 1
    if distributed:
        setup_process_group(local_rank, world_size)
    try:
        train(local_rank, world_size, args)
    finally:
        # Leave the group even if training raised, so the process exits cleanly
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--resume", action="store_true", help=f"continue from {CHECKPOINT_PATH}"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="total epochs to train (default: train.epochs in params.yaml)",
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # One process per visible GPU; CUDA_VISIBLE_DEVICES limits which ones are used
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs available: {world_size}")
    if world_size > 1:
        # Starts world_size fresh Python processes and waits for them all to finish.
        # Each one calls train_worker(i, *args), where spawn supplies i = 0..world_size-1
        # as the first argument (the local rank).
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)
    else:
        # Zero or one GPU: train in this process, with no process group and no DDP
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
