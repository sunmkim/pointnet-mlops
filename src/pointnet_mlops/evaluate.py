"""Score a trained model on the test split.

Outputs (fixed paths):
  reports/metrics.json           overall, mean per-class and per-class test accuracy
  reports/confusion_matrix.csv   rows = true class, columns = predicted class
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from pointnet_mlops.data import CLASSES, NUM_CLASSES, ModelNet10Dataset
from pointnet_mlops.model import PointNetCls
from pointnet_mlops.utils import get_device, load_params

MODEL_PATH = "models/model.pt"
METRICS_PATH = "reports/metrics.json"
CONFUSION_PATH = "reports/confusion_matrix.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model", default=MODEL_PATH, help="model file written by train.py")
    args = parser.parse_args()

    # get params
    params = load_params(args.params)
    data_params, train_params = params["data"], params["train"]
    device = get_device()
    print(f"device: {device}")

    # load test data
    test_loader = DataLoader(
        ModelNet10Dataset(data_params["prepared_dir"], "test", data_augmentation=False),
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=train_params["num_workers"],
    )

    # load model weights; build the model from the settings it was trained with, not
    # params.yaml, so an edited params file can't mismatch the saved weights
    saved_model = torch.load(args.model, map_location=device, weights_only=True)
    if tuple(saved_model["classes"]) != CLASSES:
        raise ValueError(f"{args.model} was trained on classes {saved_model['classes']}, expected {CLASSES}")
    model = PointNetCls(k=NUM_CLASSES, feature_transform=saved_model["feature_transform"]).to(device)
    model.load_state_dict(saved_model["model"])
    print(f"loaded {args.model} (epoch {saved_model['epoch']}, val acc {saved_model['val_accuracy']:.4f})")

    # set the model to evaluation mode: dropout off, BatchNorm uses its running statistics
    model.eval()

    # predict every test shape
    predictions, targets = [], []
    with torch.inference_mode():
        for points, target in test_loader:
            pred, _, _ = model(points.transpose(2, 1).to(device))
            predictions.append(pred.argmax(dim=1).cpu())
            targets.append(target)
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    # confusion[i, j] = number of class-i shapes predicted as class j
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)

    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
    metrics = {
        "accuracy": float((predictions == targets).mean()),
        "mean_class_accuracy": float(per_class_acc.mean()),
        "per_class_accuracy": {name: float(acc) for name, acc in zip(CLASSES, per_class_acc)},
        "num_test": int(len(targets)),
        "model_epoch": int(saved_model["epoch"]),
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    np.savetxt(CONFUSION_PATH, confusion, fmt="%d", delimiter=",", header=",".join(CLASSES), comments="")

    print(f"test accuracy {metrics['accuracy']:.4f}, mean class accuracy {metrics['mean_class_accuracy']:.4f}")
    for name, acc in metrics["per_class_accuracy"].items():
        print(f"  {name:<12} {acc:.4f}")
    print(f"wrote {METRICS_PATH} and {CONFUSION_PATH}")


if __name__ == "__main__":
    main()
