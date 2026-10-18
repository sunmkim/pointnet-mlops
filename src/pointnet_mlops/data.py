import os

import numpy as np
import torch
import torch.utils.data as data

# ModelNet10 class names, matching the directory names in data/ModelNet10. Label = index.
CLASSES = (
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
)
NUM_CLASSES = 10
assert len(CLASSES) == NUM_CLASSES

SPLITS = ("train", "val", "test")


class ModelNet10Dataset(data.Dataset):
    """Point clouds written by prepare.py: points (N, num_points, 3) float32, labels (N,) int64."""

    classes = CLASSES

    def __init__(self, prepared_dir, split, data_augmentation=False):
        arrays = np.load(os.path.join(prepared_dir, f"{split}.npz"))
        self.points = arrays["points"]
        self.labels = arrays["labels"]
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        point_set = self.points[index].copy()

        if self.data_augmentation:
            # ModelNet is z-up, so rotate about z (upstream rotated x/z for y-up ShapeNet)
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
            )
            point_set[:, :2] = point_set[:, :2] @ rotation_matrix  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape).astype(np.float32)  # random jitter

        return torch.from_numpy(point_set), torch.tensor(self.labels[index], dtype=torch.long)


def make_loaders(prepared_dir, batch_size, num_workers=0):
    """Train loader shuffles and augments; val/test loaders do neither."""
    loaders = {}
    for split in SPLITS:
        is_train = split == "train"
        dataset = ModelNet10Dataset(prepared_dir, split, data_augmentation=is_train)
        loaders[split] = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
        )
    return loaders
