```
uv run python -m pointnet_mlops.prepare          # only needed once to prepare your data, or after changing data params

uv run python -m pointnet_mlops.train            # fresh run
uv run python -m pointnet_mlops.train --resume   # continue after an interruption
```

## Model

[PointNet](!https://github.com/fxia22/pointnet.pytorch) from the paper ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](!https://arxiv.org/abs/1612.00593)

## Dataset

Princeton University's [ModelNet10](!https://modelnet.cs.princeton.edu/) dataset. This is a "10-class orientation-aligned subset".