from skimage import morphology
import numpy as np
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import os


class PostDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list(sorted(os.listdir(img_dir)))
        self.labels = [100, 200, 300, 400, 500, 600, 700, 800]

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = Image.open(os.path.join(self.img_dir, self.images[item]))
        return np.array(image), self.images[item]


def post_process(labels, threshold=50):

    image_dir = "../data/PCL/results"
    output_dir = "../data/PCL/results_post"
    # image_dir = "K:\dataset\遥感图像\\results"
    # output_dir = "K:\dataset\遥感图像\\results_post"
    pd = PostDataset(image_dir)
    results_loader = DataLoader(pd, batch_size=64, shuffle=False, num_workers=12)

    for image, name in results_loader:
        # print(name)
        # print(image.shape)

        for i in range(image.shape[0]):

            slice = image[i, :, :].numpy()
            for label in labels:
                new = morphology.remove_small_holes(slice == label, threshold, connectivity=1)
                slice[new] = label
            # cv2.imwrite("./test.png", slice.astype(np.uint16))
            cv2.imwrite(os.path.join(output_dir, name[i].split('.')[0]) + ".png", slice.astype(np.uint16))
            print(name[i])


if __name__ == "__main__":
    post_process(labels=[100, 200, 300, 400, 500, 600, 700, 800])