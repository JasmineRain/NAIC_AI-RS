import argparse
import numpy as np
import os
from util import semantic_to_mask, mask_to_semantic, get_confusion_matrix, get_miou
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, adamw
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import UNetPP, UNet, rf101, DANet, SEDANet, scSEUNet
from loss import lovasz_softmax, LabelSmoothSoftmaxCE, LabelSmoothCE
from utils_Deeplab import SyncBN2d
from models.DeepLabV3_plus import deeplabv3_plus
from models.HRNetOCR import seg_hrnet_ocr
from data_loader import get_dataloader


def train_val(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(img_dir=config.train_img_dir, mask_dir=config.train_mask_dir, mode="train",
                                  batch_size=config.batch_size, num_workers=config.num_workers, smooth=config.smooth)
    val_loader = get_dataloader(img_dir=config.val_img_dir, mask_dir=config.val_mask_dir, mode="val",
                                batch_size=config.batch_size, num_workers=config.num_workers)

    writer = SummaryWriter(
        comment="LR_%f_BS_%d_MODEL_%s_DATA_%s" % (config.lr, config.batch_size, config.model_type, config.data_type))

    if config.model_type not in ['UNet', 'UNet++', 'RefineNet', 'DANet', 'Deeplabv3+', 'SEDANet', 'HRNet_OCR', 'scSEUNet']:
        print('ERROR!! model_type should be selected in supported model')
        print('Choose model %s' % config.model_type)
        return
    if config.model_type == "UNet":
        model = UNet()
    elif config.model_type == "UNet++":
        model = UNetPP()
    elif config.model_type == "SEDANet":
        model = SEDANet()
    elif config.model_type == "RefineNet":
        model = rf101()
    elif config.model_type == "DANet":
        model = DANet(backbone='resnet101', nclass=config.output_ch, pretrained=True, norm_layer=SyncBN2d)
    elif config.model_type == "Deeplabv3+":
        model = deeplabv3_plus.DeepLabv3_plus(in_channels=3, num_classes=8, backend='resnet101', os=16, pretrained=True, norm_layer=SyncBN2d)
    elif config.model_type == "HRNet_OCR":
        model = seg_hrnet_ocr.get_seg_model()
    elif config.model_type == "scSEUNet":
        model = scSEUNet(pretrained=True, norm_layer=SyncBN2d)
    else:
        model = UNet()

    if config.iscontinue:
        model = torch.load("./exp/21_Deeplabv3+.pth")

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    labels = [100, 200, 300, 400, 500, 600, 700, 800]
    objects = ['水体', '交通建筑', '建筑', '耕地', '草地', '林地', '裸土', '其他']

    if config.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=1e-4, momentum=0.9)
    elif config.optimizer == "adamw":
        optimizer = adamw.AdamW(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # weight = torch.tensor([1, 1.5, 1, 2, 1.5, 2, 2, 1.2]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)

    if config.smooth == "all":
        criterion = LabelSmoothSoftmaxCE()
    elif config.smooth == "edge":
        criterion = LabelSmoothCE()
    else:
        criterion = nn.CrossEntropyLoss()

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 30, 35, 40], gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-5)

    global_step = 0
    max_miou = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        cm = np.zeros([8, 8])
        with tqdm(total=config.num_train, desc="Epoch %d / %d" % (epoch + 1, config.num_epochs),
                  unit='img') as train_pbar:
            model.train()
            for image, mask in train_loader:
                image = image.to(device, dtype=torch.float32)
                if config.smooth == "edge":
                    mask = mask.to(device, dtype=torch.float32)
                else:
                    mask = mask.to(device, dtype=torch.long).argmax(dim=1)

                pred = model(image)
                loss = criterion(pred, mask)
                # loss = lovasz_softmax(torch.softmax(pred, dim=1), mask)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                train_pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_pbar.update(image.shape[0])
                global_step += 1
                # if global_step > 10:
                #     break

            scheduler.step()
            print("training epoch loss: " + str(epoch_loss))

        val_loss = 0
        with tqdm(total=config.num_val, desc="Epoch %d / %d validation round" % (epoch + 1, config.num_epochs),
                  unit='img') as val_pbar:
            model.eval()
            locker = 0
            for image, mask in val_loader:
                image = image.to(device, dtype=torch.float32)
                target = mask.to(device, dtype=torch.long).argmax(dim=1)
                mask = mask.cpu().numpy()
                pred = model(image)
                # val_loss += lovasz_softmax(pred, target).item()
                val_loss += F.cross_entropy(pred, target).item()
                pred = pred.cpu().detach().numpy()
                mask = semantic_to_mask(mask, labels)
                pred = semantic_to_mask(pred, labels)
                cm += get_confusion_matrix(mask, pred, labels)
                val_pbar.update(image.shape[0])
                if locker == 25:
                    writer.add_images('mask_a/true', mask[2, :, :], epoch + 1, dataformats='HW')
                    writer.add_images('mask_a/pred', pred[2, :, :], epoch + 1, dataformats='HW')
                    writer.add_images('mask_b/true', mask[3, :, :], epoch + 1, dataformats='HW')
                    writer.add_images('mask_b/pred', pred[3, :, :], epoch + 1, dataformats='HW')
                locker += 1

                # break
            miou = get_miou(cm)
            if miou.mean() > max_miou:
                if torch.__version__ == "1.6.0":
                    torch.save(model.module,
                               config.result_path + "/%d_%s_%s.pth" % (epoch + 1, config.model_type, str(miou.mean())),
                               _use_new_zipfile_serialization=False)
                else:
                    torch.save(model.module,
                               config.result_path + "/%d_%s_%s.pth" % (epoch + 1, config.model_type, str(miou.mean())))
                max_miou = miou.mean()
            print(miou)
            print("testing epoch loss: " + str(val_loss))
            writer.add_scalar('miou/val', miou.mean(), epoch + 1)
            writer.add_scalar('loss/val', val_loss, epoch + 1)
            for idx, name in enumerate(objects):
                writer.add_scalar('iou/val' + name, miou[idx], epoch + 1)
    writer.close()
    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=384)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--model_type', type=str, default='scSEUNet', help='UNet/UNet++/RefineNet')
    parser.add_argument('--data_type', type=str, default='multi', help='single/multi')
    parser.add_argument('--loss', type=str, default='ce', help='ce/dice/mix')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam/adamw')
    parser.add_argument('--iscontinue', type=str, default=False, help='true/false')
    parser.add_argument('--smooth', type=str, default="edge", help='true/false')

    parser.add_argument('--train_img_dir', type=str, default="../data/PCL/train_pseudo/image")
    parser.add_argument('--train_mask_dir', type=str, default="../data/PCL/train_pseudo/mask")
    parser.add_argument('--val_img_dir', type=str, default="../data/PCL/val/image")
    parser.add_argument('--val_mask_dir', type=str, default="../data/PCL/val/mask")
    parser.add_argument('--num_train', type=int, default=190000, help="4800/1600")
    parser.add_argument('--num_val', type=int, default=10000, help="1200/400")
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--result_path', type=str, default='./exp')

    config = parser.parse_args()
    train_val(config)
