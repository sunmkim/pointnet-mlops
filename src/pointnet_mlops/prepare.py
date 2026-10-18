"""Sample point clouds from ModelNet10 meshes and write validated train/val/test arrays."""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import trimesh
import yaml
from tqdm import tqdm

from pointnet_mlops.data import CLASSES, NUM_CLASSES, SPLITS

REPORT_PATH = "reports/prepare_report.json"


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def list_meshes(raw_dir, split):
    """Return sorted (path, label) pairs for one official split ('train' or 'test')."""
    items = []
    for label, name in enumerate(CLASSES):
        split_dir = os.path.join(raw_dir, name, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"missing class directory: {split_dir}")
        for fn in sorted(os.listdir(split_dir)):
            if fn.endswith(".off"):
                items.append((os.path.join(split_dir, fn), label))
    return items


def sample_mesh(args):
    path, num_points, seed = args
    mesh = trimesh.load(path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    points = points - points.mean(axis=0)  # center
    points = points / np.max(np.linalg.norm(points, axis=1))  # scale to unit sphere
    return points.astype(np.float32)


def sample_all(items, num_points, seed_offset):
    # One seed per mesh, so the output doesn't depend on worker scheduling
    jobs = [(path, num_points, seed_offset + i) for i, (path, _) in enumerate(items)]
    with Pool() as pool:
        points = list(tqdm(pool.imap(sample_mesh, jobs, chunksize=16), total=len(jobs)))
    labels = np.array([label for _, label in items], dtype=np.int64)
    return np.stack(points), labels


def stratified_split(labels, val_fraction, seed):
    """Return (train_idx, val_idx), holding out val_fraction of each class."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(labels == label))
        n_val = max(1, round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def validate(split, points, labels, num_points):
    if points.ndim != 3 or points.shape[1:] != (num_points, 3):
        raise ValueError(f"{split}: expected shape (N, {num_points}, 3), got {points.shape}")
    if len(points) != len(labels):
        raise ValueError(f"{split}: {len(points)} point clouds but {len(labels)} labels")
    if not np.isfinite(points).all():
        raise ValueError(f"{split}: points contain NaN or inf")
    missing = sorted(set(range(NUM_CLASSES)) - set(labels.tolist()))
    if missing:
        raise ValueError(f"{split}: no samples for classes {[CLASSES[i] for i in missing]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    params = load_params(args.params)["data"]

    raw_dir = params["raw_dir"]
    out_dir = params["prepared_dir"]
    num_points = params["num_points"]
    seed = params["seed"]

    train_items = list_meshes(raw_dir, "train")
    test_items = list_meshes(raw_dir, "test")

    print(f"sampling {len(train_items)} train meshes")
    trainval_points, trainval_labels = sample_all(train_items, num_points, seed)
    print(f"sampling {len(test_items)} test meshes")
    test_points, test_labels = sample_all(test_items, num_points, seed + len(train_items))

    train_idx, val_idx = stratified_split(trainval_labels, params["val_fraction"], seed)
    splits = {
        "train": (trainval_points[train_idx], trainval_labels[train_idx]),
        "val": (trainval_points[val_idx], trainval_labels[val_idx]),
        "test": (test_points, test_labels),
    }

    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for split in SPLITS:
        points, labels = splits[split]
        validate(split, points, labels, num_points)
        np.savez(os.path.join(out_dir, f"{split}.npz"), points=points, labels=labels)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        report[split] = {
            "total": int(len(labels)),
            "per_class": {name: int(c) for name, c in zip(CLASSES, counts)},
        }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({split: report[split]["total"] for split in SPLITS}))


if __name__ == "__main__":
    main()
