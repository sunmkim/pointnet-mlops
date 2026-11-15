from ray import serve
from starlette.requests import Request
from pointnet_mlops.data import CLASSES, NUM_CLASSES
from pointnet_mlops.model import PointNetCls, feature_transform_regularizer
import torch

MODEL_PATH = "models/model.pt"


@serve.deployment
class PointNetClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load model weights
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)

        # define pytorch model from loaded weights
        self.model = PointNetCls(k=NUM_CLASSES, feature_transform=True).to(self.device)
        self.model.load_state_dict(state_dict["model"])
        
        # set the model to evaluation mode
        self.model.eval()
        
