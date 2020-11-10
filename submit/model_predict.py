import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import math
import torchvision.transforms.functional as TF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TARGET_H_DILATE = 256
TARGET_W_DILATE = 256
TARGET_H = 256
TARGET_W = 256
DILATE_SIZE = 128
DILATE = False
POST_PROCESS = False
THRESHOLD = 1536
BATCH_SIZE = 16
NUM_WORKERS = 4


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


def crop_with_dilate(img, dilate_size):
    w, h = img.size
    container = []
    target_w = TARGET_W_DILATE
    target_h = TARGET_H_DILATE

    pad_w = target_w - (w % target_w) if (w % target_w) != 0 else 0
    pad_h = target_h - (h % target_h) if (h % target_h) != 0 else 0

    # fill right and bottom
    pad_img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)

    # dilate all top/left/right/bottom
    pad_img = ImageOps.expand(pad_img, (dilate_size, dilate_size, dilate_size, dilate_size), fill=0)
    pad_img = torch.from_numpy(np.array(pad_img).astype(np.float32))
    # pad_img tensor  H W C
    # now toTensor and Normalize, final C H W tensor
    pad_img = TF.normalize((pad_img.permute(2, 0, 1) / 255), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # container: B C H W  ndarray
    for i in range(math.ceil(h / target_h)):
        for j in range(math.ceil(w / target_w)):
            crop = pad_img[:, i * target_w: (i + 1) * target_w + 2 * dilate_size,
                   j * target_h: (j + 1) * target_h + 2 * dilate_size]
            container.append(crop.numpy())
    return np.array(container), math.ceil(h / target_h), math.ceil(w / target_w)


def crop_without_dilate(img):
    # img: PIL.Image

    target_w = TARGET_W
    target_h = TARGET_H
    w, h = img.size
    container = []

    pad_w = target_w - (w % target_w) if (w % target_w) != 0 else 0
    pad_h = target_h - (h % target_h) if (h % target_h) != 0 else 0

    pad_img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)
    pad_img = torch.from_numpy(np.array(pad_img).astype(np.float32))

    # pad_img tensor  H W C
    # now toTensor and Normalize, final C H W tensor
    pad_img = TF.normalize((pad_img.permute(2, 0, 1) / 255), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # container: B C H W  ndarray
    for i in range(math.ceil(h / target_h)):
        for j in range(math.ceil(w / target_w)):
            crop = pad_img[:, i * target_w: (i + 1) * target_w, j * target_h: (j + 1) * target_h]
            container.append(crop.numpy())
    return np.array(container), math.ceil(h / target_h), math.ceil(w / target_w)


# input adarray of shape B C H W
# output final pred
def concat_without_dilate(h, w, rows, cols, pred):
    target_w = TARGET_W
    target_h = TARGET_H
    final = np.zeros((math.ceil(h / target_h) * target_h, math.ceil(w / target_w) * target_w), dtype=np.uint8)
    row, col = 0, 0
    for i in range(pred.shape[0]):
        crop = pred[i, :, :]
        final[row * target_h: (row + 1) * target_h, col * target_w: (col + 1) * target_w] = crop
        col = col + 1
        if col % cols == 0:
            row += 1
            col = 0
    final = final[0: h, 0: w]

    return final


def concat_with_dilate(h, w, rows, cols, pred, dilate_size):
    target_w = TARGET_W_DILATE
    target_h = TARGET_H_DILATE
    final = np.zeros((math.ceil(h / target_h) * target_h, math.ceil(w / target_w) * target_w), dtype=np.uint8)
    row, col = 0, 0
    for i in range(pred.shape[0]):
        crop = pred[i, dilate_size:dilate_size + target_h, dilate_size:dilate_size + target_w]
        final[row * target_h: (row + 1) * target_h, col * target_w: (col + 1) * target_w] = crop
        col = col + 1
        if col % cols == 0:
            row += 1
            col = 0
    final = final[0: h, 0: w]

    return final


def predict(models, input_path, output_dir):
    with torch.no_grad():

        name, ext = os.path.splitext(input_path)
        name = os.path.split(name)[-1] + ".png"
        image = Image.open(input_path)
        w, h = image.size
        labels = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

        threshold = THRESHOLD
        dilate_size = DILATE_SIZE
        dilate = DILATE

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if dilate:
            cropped, rows, cols = crop_with_dilate(image, dilate_size)
        else:
            cropped, rows, cols = crop_without_dilate(image)

        if h < threshold and w < threshold:
            image = torch.from_numpy(cropped).to(device, dtype=torch.float32)
            total_pred = np.zeros((image.shape[0], len(labels), image.shape[-2], image.shape[-1]))
            for model in models:
                pred = model(image)
                total_pred += pred.cpu().detach().numpy()
            pred = total_pred
            # after to_mask pred shape: B H W
            pred = semantic_to_mask(pred, labels=labels)
        else:
            dataset = RSDataset(cropped)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            all_pred = None
            for images in dataloader:
                total_pred = np.zeros((image.shape[0], len(labels), image.shape[-2], image.shape[-1]))
                images = images.to(device, dtype=torch.float32)
                for model in models:
                    pred = model(images)
                    total_pred += pred.cpu().detach().numpy()
                pred = total_pred
                pred = semantic_to_mask(pred, labels=labels)
                all_pred = np.concatenate((all_pred, pred), axis=0) if all_pred is not None else pred
            pred = all_pred

        if dilate:
            final = concat_with_dilate(h, w, rows, cols, pred, dilate_size)
        else:
            final = concat_without_dilate(h, w, rows, cols, pred)

        cv2.imwrite(os.path.join(output_dir, name), final)
