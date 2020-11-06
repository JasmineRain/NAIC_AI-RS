import os
import torch
from models import DANet
import torch.nn as nn
from models.DeepLabV3_plus import deeplabv3_plus


@torch.no_grad()
def restore_models():

    src = "./exp/45_Deeplabv3+_0.7594.pth"
    model = torch.load(src, map_location="cpu").module
    print(model)
    state_dict = model.state_dict()
    torch.save(state_dict, "./model.pth")


if __name__ == "__main__":
    restore_models()