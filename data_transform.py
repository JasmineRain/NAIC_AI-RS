import torchvision.transforms.functional as F
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import PIL
from util import mask_to_semantic


# class CenterCrop(object):
#
#     def __init__(self, size=256):
#         self.size = size
#
#     def __call__(self, image, mask):
#
#         # image transform
#         image_0 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 0]), output_size=self.size)), axis=2)
#         image_1 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 1]), output_size=self.size)), axis=2)
#         image_2 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 2]), output_size=self.size)), axis=2)
#         image = np.concatenate((image_0, image_1, image_2), axis=2)
#
#         # mask transform
#         mask = np.array(F.center_crop(img=Image.fromarray(mask), output_size=self.size))
#
#         return image, mask


class HorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return image, mask


class VerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return image, mask


class Rotate(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):

        angle = random.choice(self.degrees)

        if angle == 90:
            image = image.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        elif angle == 180:
            image = image.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)
        elif angle == 270:
            image = image.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)

        return image, mask


class Resize(object):
    def __init__(self, p=0.5, scales=[(320, 320), (192, 192), (384, 384), (128, 128)]):
        self.scales = scales
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            scale = random.choice(self.scales)
            image = image.resize(scale, resample=PIL.Image.BILINEAR)
            mask = mask.resize(scale, resample=PIL.Image.BILINEAR)

        return image, mask


class Brighten(object):

    def __init__(self, p=0.5, alpha=1.3):
        self.p = p
        self.alpha = alpha

    def __call__(self, image):

        if random.random() < self.p:
            en = ImageEnhance.Brightness(image)
            image = en.enhance(self.alpha)

        return image


class Color(object):
    def __init__(self, p=0.5, alpha=1.5):
        self.p = p
        self.alpha = alpha

    def __call__(self, image):

        if random.random() < self.p:
            en = ImageEnhance.Color(image)
            image = en.enhance(self.alpha)

        return image


class Contrast(object):
    def __init__(self, p=0.5, alpha=1.5):
        self.p = p
        self.alpha = alpha

    def __call__(self, image):

        if random.random() < self.p:
            en = ImageEnhance.Contrast(image)
            image = en.enhance(self.alpha)

        return image


class GaussianBlur(object):

    def __init__(self, p=0.5, radius=1.5):
        self.p = p
        self.radius = radius

    def __call__(self, image):
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return image


class ToTensor(object):

    def __call__(self, image, mask, labels=[100, 200, 300, 400, 500, 600, 700, 800], mode="train"):
        # image transform

        image = np.array(image).astype(np.float)
        mask = np.array(mask)

        if mode == "test":
            for i in range(image.shape[2]):
                image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (
                            np.max(image[:, :, i]) - np.min(image[:, :, i]))
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            return image

        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i]) - np.min(image[:, :, i]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # mask transform to semantic
        mask = torch.from_numpy(mask_to_semantic(mask, labels).transpose((2, 0, 1)))
        return image, mask


class Normalize(object):

    def __call__(self, image, mask, mode="train"):

        # image transform

        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mode == "test":
            return image

        return image, mask