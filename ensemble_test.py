import numpy as np
from PIL import Image
import os
from sys import exit

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torchvision.transforms.functional as F
from util import semantic_to_mask
import torch
import cv2
from data_loader import get_dataloader
from models import DANet

def rot(input, k, degree):
    if degree not in [90, 180, 270]:
        raise Exception
    for i in range(degree // 90):
        input = torch.rot90(input, k, dims=[input.dim() - 2, input.dim() - 1])
    return input

@torch.no_grad()
def generate_test():
    batch_size = 128

    # input = "../data/PCL/image_B"
    # input_dir = "../../ly/data/PCL/image_A"
    # mask_dir = "../../ly/data/PCL/image_A"
    input_dir = "../data/PCL/image_B"
    mask_dir = "../data/PCL/image_B"
    output_dir = "../data/PCL/results"
    models_dir = './last_try_best'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images = list(sorted(os.listdir(input_dir)))
    models_path = [os.path.join(models_dir, name) for name in os.listdir(models_dir)]
    models = []
    for model_path in models_path:
        # if model_path.endswith('79.90.pth'):
            # continue
        model = torch.load(model_path).module
        model = model.to(device)
        model.eval()
        models.append(model)
        print(model_path)

    print('ensemble %d models!!' % len(models))

    labels = [100, 200, 300, 400, 500, 600, 700, 800]

    # for image_name in images:
    #     print(image_name)
    #     image = np.array(Image.open(os.path.join(input, image_name))).astype(float)
    #     for i in range(image.shape[2]):
    #         image[:, :, i] = (image[:, :, i] - image[:, :, i].min()) / (image[:, :, i].max() - image[:, :, i].min())
    #     image = torch.from_numpy(image.transpose((2, 0, 1)))
    #     image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     image = torch.unsqueeze(image, 0)
    #     image = image.to(device, dtype=torch.float32)
    #     pred = torch.softmax(model(image), dim=1).cpu().detach().numpy()
    #     pred = semantic_to_mask(pred, labels).squeeze().astype(np.uint16)
    #     cv2.imwrite(os.path.join(output, image_name.split('.')[0]) + ".png", pred)

    test_loader = get_dataloader(img_dir=input_dir, mask_dir=mask_dir, mode="test",
                                 batch_size=batch_size, num_workers=8)

    for image, _, name in test_loader:
        # print(name)
        # print(image.shape)
        total_pred = np.zeros((image.shape[0], len(labels), 256, 256))
        image = image.to(device, dtype=torch.float32)
        # print(type(image))
        # print(image.shape)

        for i,model in enumerate(models):
            # 测试集增强，逆时针旋转90、180、270，上下翻转，左右翻转
            total_pred_transform = np.zeros((image.shape[0], len(labels), 256, 256))
            # transform_funcs = [rot, rot, rot, torch.flip, torch.flip]
            # transform_params = [(1, 90), (1, 180), (1, 270), ([image.dim() - 2],), ([image.dim() - 1],)]
            # recover_params = [(-1, 90), (-1, 180), (-1, 270), ([total_pred_transform.ndim - 2],),
            #                   ([total_pred_transform.ndim - 1],)]

            output = model(image)
            pred = torch.softmax(output, dim=1).cpu().detach().numpy()
            total_pred_transform += pred

            # for i, transform in enumerate(transform_funcs):
            #     image_transform = transform(*((image,) + transform_params[i]))
            #     output = model(image_transform)
            #     pred = torch.softmax(output, dim=1).cpu().detach()
            #     pred = transform(*((pred,) + recover_params[i])).numpy()  # 复原
            #     total_pred_transform += pred
            # total_pred_transform /= (len(transform_funcs) + 1)

            total_pred += total_pred_transform

        total_pred /= len(models)
        total_pred = semantic_to_mask(total_pred, labels=labels).squeeze().astype(np.uint16)
        # print(total_pred.shape)
        # print(pred.shape, pred.dtype)
        for i in range(total_pred.shape[0]):
            cv2.imwrite(os.path.join(output_dir, name[i].split('.')[0]) + ".png", total_pred[i, :, :])
            print(name[i])

if __name__ == "__main__":
    generate_test()

