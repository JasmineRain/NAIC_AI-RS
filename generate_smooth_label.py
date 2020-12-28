import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from util import mask_to_semantic


class PostDataset(Dataset):
    def __init__(self, mask_dir):
        self.mask_dir = mask_dir
        self.masks = list(sorted(os.listdir(mask_dir)))
        self.labels = [100, 200, 300, 400, 500, 600, 700, 800]

    def __len__(self):
        return len(os.listdir(self.mask_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        mask = Image.open(os.path.join(self.mask_dir, self.masks[item]))
        return np.array(mask), self.masks[item]


def generate_smooth_label(labels=[100, 200, 300, 400, 500, 600, 700, 800], alpha=0.2, radius=5):

    # label 标签 [100, 200, 300, 400, 500, 600, 700, 800]
    # alpha 平滑系数 0.2
    # radius 平滑半径 8

    mask_dir = "../data/PCL/train_pseudo/mask"
    output_dir = "../data/PCL/train_pseudo/mask_smooth"
    mask_dir = "K:\dataset\遥感图像\\results"
    output_dir = "K:\dataset\遥感图像\\results_post"
    pd = PostDataset(mask_dir)
    results_loader = DataLoader(pd, batch_size=16, shuffle=False, num_workers=4)

    for mask, name in results_loader:

        for i in range(mask.shape[0]):

            slice = mask[i, :, :]
            semantic_map = mask_to_semantic(slice.numpy(), labels, smooth='edge', alpha=alpha, radius=radius)
            np.save(os.path.join(output_dir, name[i].split('.')[0]) + ".npy", semantic_map)
            print(name[i])


if __name__ == "__main__":
    generate_smooth_label()