import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from util import semantic_to_mask
import torch
import cv2
from data_loader import get_dataloader
import torch.nn as nn


@torch.no_grad()
def generate_test():

    output_dir = "../data/PCL/results"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load("./exp/22_DANet.pth").module

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    labels = [100, 200, 300, 400, 500, 600, 700, 800]

    test_loader = get_dataloader(img_dir="../data/PCL/image_A", mask_dir="../data/PCL/image_A", mode="test",
                                  batch_size=64, num_workers=8)

    for image, _, name in test_loader:
        image = image.to(device, dtype=torch.float32)
        output = model(image)
        pred = torch.softmax(output, dim=1).cpu().detach().numpy()
        pred = semantic_to_mask(pred, labels=labels).squeeze().astype(np.uint16)
        for i in range(pred.shape[0]):
            cv2.imwrite(os.path.join(output_dir, name[i].split('.')[0]) + ".png", pred[i, :, :])
            print(name[i])


if __name__ == "__main__":
    generate_test()
    exit(0)
