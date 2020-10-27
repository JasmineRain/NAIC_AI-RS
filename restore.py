import os
import torch
from models import DANet
import torch.nn as nn
from models.DeepLabV3_plus import deeplabv3_plus
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


@torch.no_grad()
def restore_models():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    src = "./pretrained/Deeplabv3+.pth"
    pretrained_dict = torch.load(src, map_location='cpu').module.state_dict()
    # print(pretrained_dict.keys())
    for key in list(pretrained_dict.keys()):
        if key.split('.')[0] == "cbr_last":
            pretrained_dict.pop(key)
    model = deeplabv3_plus.DeepLabv3_plus(in_channels=3, num_classes=15, backend='resnet101', os=16, pretrained=True, norm_layer=nn.BatchNorm2d)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    image = torch.rand(1, 3, 5000, 5000, device=device)
    output = model(image)
    print(output.shape)


if __name__ == "__main__":
    restore_models()