from skimage import morphology
import numpy as np
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import exit

plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置


class PostDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list(sorted(os.listdir(img_dir)))
        self.labels = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = Image.open(os.path.join(self.img_dir, self.images[item]))
        return np.array(image), self.images[item]


def visual_img():
    # image_dir = "../data/PCL/results_pre"
    image_dir = "out"
    output_dir = "color"
    # image_dir = "K:\dataset\遥感图像\\results"
    # output_dir = "K:\dataset\遥感图像\\results_post"

    # 天蓝,红,     蓝,  屎黄, 浅绿, 深绿, 灰白, 紫色
    # 水体,交通建筑,建筑,耕地, 草地, 林地, 裸土, 其他
    colors = np.array(
        [[255, 127, 39], [255, 202, 24], [253, 236, 166], [196, 255, 14], [14, 209, 69],
         [140, 255, 251], [0, 168, 243], [63, 72, 204], [184, 61, 186], [185, 122, 86],
         [195, 195, 195], [236, 28, 36], [136, 0, 27], [88, 88, 88], [255, 174, 200]])
    pd = PostDataset(image_dir)
    results_loader = DataLoader(pd, batch_size=1, shuffle=False, num_workers=12)
    labels = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    for images, names in results_loader:
        images = images.numpy()
        images = np.expand_dims(images, axis=0).repeat(3, axis=0)
        vivid_imgs = np.zeros_like(images)
        print(vivid_imgs.shape)
        for i, label in enumerate(labels):
            # vivid_imgs[images == label] = colors[i][::-1].repeat(len(vivid_imgs[images == label]) // len(colors[i]))
            vivid_imgs[images == label] = colors[i].repeat(len(vivid_imgs[images == label]) // len(colors[i]))

        # vivid_imgs = np.uint8(vivid_imgs)

        vivid_imgs = vivid_imgs.transpose((1, 2, 3, 0))

        for i, image in enumerate(vivid_imgs):
            rgb = colors
            rgb = np.array(rgb) / 255.0
            icmap = mpl.colors.ListedColormap(rgb, name='my_color')
            norm = mpl.colors.Normalize(vmin=0, vmax=15)
            fig = plt.figure(figsize=(10, 8))
            h = plt.imshow(image, cmap=icmap, norm=norm)
            cbar = plt.colorbar(mappable=h)
            v = np.linspace(0, 14, 15)
            cbar.set_ticks((v + 0.5))
            cbar.set_ticklabels(['水体', '道路', '建筑物', '机场', '停车场', '操场', '普通耕地', '农业大棚', '自然草地', '绿地绿化',
                                 '自然林', '人工林', '自然裸土', '人为裸土', '其它'])
            # plt.show()
            plt.savefig(os.path.join(output_dir, names[i].split('.')[0]) + ".png")
            # cv2.imwrite(os.path.join(output_dir, names[i].split('.')[0]) + ".png", image)
            plt.close()
            print(names[i])


if __name__ == "__main__":
    visual_img()
