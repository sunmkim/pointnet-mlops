

## Model

[PointNet](!https://github.com/fxia22/pointnet.pytorch) from the paper ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](!https://arxiv.org/abs/1612.00593)

## Dataset

Princeton University's [ModelNet10](!https://modelnet.cs.princeton.edu/) dataset. This is a "10-class orientation-aligned subset".

## Training and Evaluation
### Data Preparation
To prepare the data, run this command below. This is needed only once to prepare the data, or after changing data params
```
uv run python -m pointnet_mlops.prepare
```

### Model Training
```
uv run python -m pointnet_mlops.train            # fresh run
uv run python -m pointnet_mlops.train --resume   # continue after an interruption
```

### Model Evals
```
uv run python -m pointnet_mlops.evaluate
uv run python -m pointnet_mlops.evaluate --model path/to/model.pt   # re-score another model
```
