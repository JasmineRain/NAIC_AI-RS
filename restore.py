import os
import torch
from models import DANet
import torch.nn as nn
from models.DeepLabV3_plus import deeplabv3_plus


@torch.no_grad()
def restore_models():

    src = "./exp/8_HRNet_OCR_0.7314.pth"
    model = torch.load(src, map_location="cpu").module
    print(model)
    state_dict = model.state_dict()
    torch.save(state_dict, "./model_hrnet.pth")


if __name__ == "__main__":
    restore_models()