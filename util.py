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


def label_smooth(mask, semantic_map, labels, alpha, radius):
    """
    :param
        mask: mask of shape [H, W]
        semantic_map: one-hot mask of shape [C, H, W]
    """
    pad_mask = np.pad(mask, ((radius, radius), (radius, radius)), 'edge')

    for i in range(radius, len(pad_mask[0]) - radius):
        for j in range(radius, len(pad_mask[0]) - radius):
            if not (pad_mask[i][j] == pad_mask[i - radius: i + radius + 2, j - radius: j + radius + 2]).all():
                pos = 1. - alpha
                neg = alpha / len(labels)
                idx = semantic_map[:, i - radius, j - radius].argmax()
                semantic_map[:, i - radius, j - radius].fill(neg)
                semantic_map[:, i - radius, j - radius][idx] = pos + neg
    return semantic_map


def mask_to_semantic(mask, labels=[100, 200, 300, 400, 500, 600, 700, 800], smooth=True):
    # 标签图转语义图  labels代表所有的标签 [100, 200, 300, 400, 500, 600, 700, 800]
    # return shape [C, H, W]

    semantic_map = []
    for label in labels:
        equality = np.equal(mask, label)
        semantic_map.append(equality)
    semantic_map = np.array(semantic_map).astype(np.float16)
    if smooth:
        semantic_map = label_smooth(mask, semantic_map, labels, alpha=0.1, radius=2)

    return semantic_map


def semantic_to_mask(mask, labels):
    # 语义图转标签图  labels代表所有的标签 [100, 200, 300, 400, 500, 600, 700, 800]
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
    # generate_label()
    # pred = np.array([[100, 200, 300],
    #                  [400, 500, 600],
    #                  [100, 100, 100]])
    #
    label = np.array([[100, 200, 300],
                      [400, 500, 600],
                      [700, 800, 100]])

    # print(label.fill(0))
    # print(label)
    results = mask_to_semantic(label, smooth=True)
    print(results[:, 2, 1])

