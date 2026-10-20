import torch

from pointnet_mlops.data import NUM_CLASSES
from pointnet_mlops.model import PointNetCls


def main():
    model = PointNetCls(k=NUM_CLASSES, feature_transform=feature_transform).to(device)

    model.load_state_dict(
        torch.load("model.pt", map_location=device, weights_only=True)
    )


if __name__ == "__main__":
    main()
