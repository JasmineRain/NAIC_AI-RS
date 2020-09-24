import numpy as np
from PIL import Image
import os
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(true, pred, labels):

    true = true.flatten()
    pred = pred.flatten()

    return confusion_matrix(true, pred, labels)


def get_miou(cm):
    return np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))


def mask_to_semantic(mask, labels):
    semantic_map = []
    for label in labels:
        equality = np.equal(mask, label)
        semantic_map.append(equality)
    semantic_map = np.array(semantic_map).transpose((1, 2, 0)).astype(np.uint8)
    return semantic_map


def semantic_to_mask(mask, labels):
    x = np.argmax(mask, axis=1)
    label_codes = np.array(labels)
    x = np.uint16(label_codes[x.astype(np.uint8)])
    return x


def generate_label():
    path = "./val/mask"
    output = "./val/label"
    # path = "./mask"
    # output = "./label"
    labels = [100, 200, 300, 400, 500, 600, 700, 800]
    files = os.listdir(path)
    for file in files:
        mask = np.array(Image.open(os.path.join(path, file)))
        mask = mask_to_semantic(mask, labels)
        print(mask.shape)
        np.save(os.path.join(output, file.split(".")[0]), mask)


# def generate_mask():
#     path = "../data/PCL/train/mask"
#     output = "./data/PCL/train/label"
#     labels = [100, 200, 300, 400, 500, 600, 700, 800]
#     files = os.listdir(path)
#     for file in files:
#         mask = np.array(Image.open(os.path.join(path, file)))
#         mask = mask_to_semantic(mask, labels)
#         np.save(os.path.join(output, file), mask)

if __name__ == "__main__":

    generate_label()
    # pred = np.array([[100, 200, 300],
    #                  [400, 500, 600],
    #                  [100, 100, 100]])
    #
    label = np.array([[100, 200, 300],
                      [400, 500, 600],
                      [700, 800, 100]])
    #


