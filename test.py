import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torchvision.transforms.functional as F
from util import semantic_to_mask
import torch
import cv2
from data_loader import get_dataloader
from models import DANet

@torch.no_grad()
def generate_test():

    input = "../data/PCL/image_A"
    output_dir = "../data/PCL/results"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images = list(sorted(os.listdir(input)))
    model = torch.load("./exp/22_DANet.pth").module
    model = model.to(device)
    model.eval()

    labels = [100, 200, 300, 400, 500, 600, 700, 800]

    # for image_name in images:
    #     print(image_name)
    #     image = np.array(Image.open(os.path.join(input, image_name))).astype(float)
    #     for i in range(image.shape[2]):
    #         image[:, :, i] = (image[:, :, i] - image[:, :, i].min()) / (image[:, :, i].max() - image[:, :, i].min())
    #     image = torch.from_numpy(image.transpose((2, 0, 1)))
    #     image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     image = torch.unsqueeze(image, 0)
    #     image = image.to(device, dtype=torch.float32)
    #     pred = torch.softmax(model(image), dim=1).cpu().detach().numpy()
    #     pred = semantic_to_mask(pred, labels).squeeze().astype(np.uint16)
    #     cv2.imwrite(os.path.join(output, image_name.split('.')[0]) + ".png", pred)

    test_loader = get_dataloader(img_dir="../data/PCL/image_A", mask_dir="../data/PCL/image_A", mode="test",
                                  batch_size=64, num_workers=8)

    for image, _, name in test_loader:
        # print(name)
        # print(image.shape)
        image = image.to(device, dtype=torch.float32)
        output = model(image)
        pred = torch.softmax(output, dim=1).cpu().detach().numpy()
        pred = semantic_to_mask(pred, labels=labels).squeeze().astype(np.uint16)
        # print(pred.shape)
        # print(pred.shape, pred.dtype)
        for i in range(pred.shape[0]):
            cv2.imwrite(os.path.join(output_dir, name[i].split('.')[0]) + ".png", pred[i, :, :])
            print(name[i])


if __name__ == "__main__":
    generate_test()
    exit(0)
