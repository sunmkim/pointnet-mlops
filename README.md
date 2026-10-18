```
uv run python -m pointnet_mlops.prepare     # only needed once, or after changing data params
uv run python -m pointnet_mlops.train --nepoch 1
```

## Model

[PointNet](!https://github.com/fxia22/pointnet.pytorch) from the paper ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](!https://arxiv.org/abs/1612.00593)

## Dataset

Princeton University's [ModelNet10](!https://modelnet.cs.princeton.edu/) dataset. This is a "10-class orientation-aligned subset".