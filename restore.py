import numpy as np
from PIL import Image
import os
import torchvision.transforms.functional as F
from util import semantic_to_mask
import torch
import cv2
from data_loader import get_dataloader
from models import DANet

def restore_models():

    src = "./ensemble"
    names = os.listdir(src)

    for name in names:
        model = torch.load(os.path.join(src, name)).module
        torch.save(model, "./ensemble/new_%s" % name, _use_new_zipfile_serialization=False)
        print(name + " has been restored")


if __name__ == "__main__":
    restore_models()