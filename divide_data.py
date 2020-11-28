import numpy as np
import os
import shutil
import random


def divede():

    train_img_dir = "./data/2020_naic_remote_sensing/semi_final/train/images"
    train_mask_dir = "./data/2020_naic_remote_sensing/semi_final/train/labels"

    train_image_output = "./train_data/image"
    train_mask_output = "./train_data/mask"
    val_image_output = "./val_data/image"
    val_mask_output = "./val_data/mask"

    files = list(sorted(os.listdir(train_img_dir)))

    val_files_idx = random.sample(range(0, 100000), 1000)

    print("start creating dataset...")

    for idx in range(100000):
        if idx in val_files_idx:
            shutil.copy(os.path.join(train_img_dir, files[idx]), os.path.join(val_image_output, files[idx]))
            shutil.copy(os.path.join(train_mask_dir, files[idx].split('.')[0] + ".png"), os.path.join(val_mask_output, files[idx].split('.')[0] + ".png"))
        else:
            shutil.copy(os.path.join(train_img_dir, files[idx]), os.path.join(train_image_output, files[idx]))
            shutil.copy(os.path.join(train_mask_dir, files[idx].split('.')[0] + ".png"), os.path.join(train_mask_output, files[idx].split('.')[0] + ".png"))


def merge():
    train_img_dir = "./extra_data/images"
    train_mask_dir = "./extra_data/mask"

    train_image_output = "./train_data/image"
    train_mask_output = "./train_data/mask"

    print("start merging dataset...")

    files = os.listdir(train_img_dir)
    for file in files:
        shutil.copy(os.path.join(train_img_dir, file), os.path.join(train_image_output, file))

    files = os.listdir(train_mask_dir)
    for file in files:
        shutil.copy(os.path.join(train_mask_dir, file), os.path.join(train_mask_output, file))


if __name__ == "__main__":
    divede()
    print("creating dataset finished")
    merge()
    print("merging dataset finished")
