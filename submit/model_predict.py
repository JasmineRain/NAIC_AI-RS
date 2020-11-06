import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as NF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class RSDataset(Dataset):

    def __init__(self, cropped):
        self.images = cropped

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item, :, :, :]
        return image


def semantic_to_mask(mask, labels):
    x = np.argmax(mask, axis=1)
    label_codes = np.array(labels)
    x = np.uint8(label_codes[x.astype(np.uint8)])
    return x


def collect_cropped_images(container, img):
    container.append(img)


def crop_with_dilate(img):
    return img


def crop_without_dilate(img):
    # img: PIL.Image

    target_w = target_h = 256
    w, h = img.size
    container = []

    if w % target_w != 0 and h % target_h != 0:
        pad_w = target_w - (w % target_w)
        pad_h = target_h - (h % target_h)
        pad_img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)
        pad_img = torch.from_numpy(np.array(pad_img).astype(np.float32))
    else:
        pad_img = torch.from_numpy(np.array(img).astype(np.float32))

    # pad_img tensor  H W C
    # now toTensor and Normalize, final C H W tensor
    pad_img = TF.normalize((pad_img.permute(2, 0, 1) / 255), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # container: B C H W  ndarray
    for i in range(math.ceil(h / target_h)):
        for j in range(math.ceil(w / target_w)):
            crop = pad_img[:, i * target_w: (i + 1) * target_w, j * target_h: (j + 1) * target_h]
            container.append(crop.numpy())
    return np.array(container), math.ceil(h / target_h), math.ceil(w / target_w)


def predict(model, input_path, output_dir):
    with torch.no_grad():
        # start_time = time.time()
        name, ext = os.path.splitext(input_path)
        name = os.path.split(name)[-1] + ".png"
        image = Image.open(input_path)
        w, h = image.size
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if h < 1536 and w < 1536:

            # cropped shape: B H W C
            cropped, rows, cols = crop_without_dilate(image)
            image = torch.from_numpy(cropped).to(device, dtype=torch.float32)
            pred = model(image)
            pred = pred.cpu().detach().numpy()

            # after to_mask pred shape: B H W
            pred = semantic_to_mask(pred, labels=[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
            final = np.zeros((math.ceil(h / 256) * 256, math.ceil(w / 256) * 256), dtype=np.uint8)
            row, col = 0, 0
            for i in range(pred.shape[0]):
                crop = pred[i, :, :]
                final[row * 256: (row + 1) * 256, col * 256: (col + 1) * 256] = crop
                col = col + 1
                if col % cols == 0:
                    row += 1
                    col = 0
            final = final[0: h, 0: w]
            cv2.imwrite(os.path.join(output_dir, name), final)

        else:
            cropped, rows, cols = crop_without_dilate(image)
            dataset = RSDataset(cropped)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

            final = np.zeros((math.ceil(h / 256) * 256, math.ceil(w / 256) * 256), dtype=np.uint8)
            row, col = 0, 0
            for images in dataloader:
                images = images.to(device, dtype=torch.float32)
                pred = model(images)
                pred = pred.cpu().detach().numpy()
                pred = semantic_to_mask(pred, labels=[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
                for i in range(pred.shape[0]):
                    crop = pred[i, :, :]
                    final[row * 256: (row + 1) * 256, col * 256: (col + 1) * 256] = crop
                    col += 1
                    if col % cols == 0:
                        row += 1
                        col = 0
            final = final[0: h, 0: w]
            cv2.imwrite(os.path.join(output_dir, name), final)
