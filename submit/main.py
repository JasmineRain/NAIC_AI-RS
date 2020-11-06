import os
import importlib
from model_define import init_model
from model_predict import predict
import numpy as np
from PIL import Image


def generate_outputs(src, output_dir):

    labels = np.array([1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    flag = True

    files = list(os.listdir(src))
    input_paths = []
    for file in files:
        input_paths.append(os.path.join(src, file))

    model = init_model()
    for input_path in input_paths:
        predict(model, input_path, output_dir)

    print("start checking file: ")
    for file in files:
        name = file.split('.')[0]
        image = np.array(Image.open("./robust_test/" + name + ".tiff"))
        pred = np.array(Image.open("./robust_out/" + name + ".png"))

        if not (image.shape[:2] == pred.shape):
            flag = False
            print("wrong output shape! ", flag)
            return

        pred_labels = np.unique(pred)
        if np.setdiff1d(pred_labels, labels).size > 0:
            flag = False
            print("wrong output label! ", flag, pred_labels, name)
            return
    print("check finished!", flag)


if __name__ == "__main__":
    generate_outputs("./input", "./out")
