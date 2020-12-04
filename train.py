import argparse
import numpy as np
import os
from util import semantic_to_mask, mask_to_semantic, get_confusion_matrix, get_miou
from utils_Deeplab import SyncBN2d
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
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

    if config.model_type == "UNet":
        model = UNet()
    elif config.model_type == "UNet++":
        model = UNetPP()
    elif config.model_type == "SEDANet":
        model = SEDANet()
    elif config.model_type == "RefineNet":
        model = rf101()
    elif config.model_type == "DANet":
        # src = "./pretrained/60_DANet_0.8086.pth"
        # pretrained_dict = torch.load(src, map_location='cpu').module.state_dict()
        # print("load pretrained params from stage 1: " + src)
        # pretrained_dict.pop('seg1.1.weight')
        # pretrained_dict.pop('seg1.1.bias')
        model = DANet(backbone='resnext101', nclass=config.output_ch, pretrained=True, norm_layer=nn.BatchNorm2d)
        # model_dict = model.state_dict()
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    elif config.model_type == "Deeplabv3+":
        # src = "./pretrained/Deeplabv3+.pth"
        # pretrained_dict = torch.load(src, map_location='cpu').module.state_dict()
        # print("load pretrained params from stage 1: " + src)
        # # print(pretrained_dict.keys())
        # for key in list(pretrained_dict.keys()):
        #     if key.split('.')[0] == "cbr_last":
        #         pretrained_dict.pop(key)
        model = deeplabv3_plus.DeepLabv3_plus(in_channels=3, num_classes=config.output_ch, backend='resnet101', os=16, pretrained=True, norm_layer=nn.BatchNorm2d)
        # model_dict = model.state_dict()
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    elif config.model_type == "HRNet_OCR":
        model = seg_hrnet_ocr.get_seg_model()
    elif config.model_type == "scSEUNet":
        model = scSEUNet(pretrained=True, norm_layer=nn.BatchNorm2d)
    else:
        model = UNet()

    if config.iscontinue:
        model = torch.load("./exp/13_Deeplabv3+_0.7619.pth", map_location='cpu').module

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    labels = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    objects = ['水体', '道路', '建筑物', '机场', '停车场', '操场', '普通耕地', '农业大棚', '自然草地', '绿地绿化',
               '自然林', '人工林', '自然裸土', '人为裸土', '其它']
    frequency = np.array([0.0279, 0.0797, 0.1241, 0.00001, 0.0616, 0.0029, 0.2298, 0.0107, 0.1207, 0.0249,
                          0.1470, 0.0777, 0.0617, 0.0118, 0.0187])

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
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5, verbose=True)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-4)

    global_step = 0
    max_fwiou = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        seed = np.random.randint(0, 2, 1)
        seed = 0
        print("seed is ", seed)
        if seed == 1:
            train_loader = get_dataloader(img_dir=config.train_img_dir, mask_dir=config.train_mask_dir, mode="train",
                                          batch_size=config.batch_size // 2, num_workers=config.num_workers,
                                          smooth=config.smooth)
            val_loader = get_dataloader(img_dir=config.val_img_dir, mask_dir=config.val_mask_dir, mode="val",
                                        batch_size=config.batch_size // 2, num_workers=config.num_workers)
        else:
            train_loader = get_dataloader(img_dir=config.train_img_dir, mask_dir=config.train_mask_dir, mode="train",
                                          batch_size=config.batch_size, num_workers=config.num_workers,
                                          smooth=config.smooth)
            val_loader = get_dataloader(img_dir=config.val_img_dir, mask_dir=config.val_mask_dir, mode="val",
                                        batch_size=config.batch_size, num_workers=config.num_workers)

        cm = np.zeros([15, 15])
        print(optimizer.param_groups[0]['lr'])
        with tqdm(total=config.num_train, desc="Epoch %d / %d" % (epoch + 1, config.num_epochs),
                  unit='img', ncols=100) as train_pbar:
            model.train()

            for image, mask in train_loader:
                image = image.to(device, dtype=torch.float32)

                if seed == 0:
                    pass
                elif seed == 1:
                    image = F.interpolate(image, size=(384, 384), mode='bilinear', align_corners=True)
                    mask = F.interpolate(mask.float(), size=(384, 384), mode='nearest')

                if config.smooth == "edge":
                    mask = mask.to(device, dtype=torch.float32)
                else:
                    mask = mask.to(device, dtype=torch.long).argmax(dim=1)

                aux_out, out = model(image)
                aux_loss = criterion(aux_out, mask)
                seg_loss = criterion(out, mask)
                loss = aux_loss + seg_loss

                # pred = model(image)
                # loss = criterion(pred, mask)

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

            # scheduler.step()
            print("\ntraining epoch loss: " + str(epoch_loss / (float(config.num_train) / (float(config.batch_size)))))
        torch.cuda.empty_cache()
        val_loss = 0
        with torch.no_grad():
            with tqdm(total=config.num_val, desc="Epoch %d / %d validation round" % (epoch + 1, config.num_epochs),
                      unit='img', ncols=100) as val_pbar:
                model.eval()
                locker = 0
                for image, mask in val_loader:
                    image = image.to(device, dtype=torch.float32)
                    target = mask.to(device, dtype=torch.long).argmax(dim=1)
                    mask = mask.cpu().numpy()
                    _, pred = model(image)
                    val_loss += F.cross_entropy(pred, target).item()
                    pred = pred.cpu().detach().numpy()
                    mask = semantic_to_mask(mask, labels)
                    pred = semantic_to_mask(pred, labels)
                    cm += get_confusion_matrix(mask, pred, labels)
                    val_pbar.update(image.shape[0])
                    if locker == 5:
                        writer.add_images('mask_a/true', mask[2, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_a/pred', pred[2, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_b/true', mask[3, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_b/pred', pred[3, :, :], epoch + 1, dataformats='HW')
                    locker += 1

                    # break
                miou = get_miou(cm)
                fw_miou = (miou * frequency).sum()
                scheduler.step()

                if True:
                    if torch.__version__ == "1.6.0":
                        torch.save(model,
                                   config.result_path + "/%d_%s_%.4f.pth" % (epoch + 1, config.model_type, fw_miou),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model,
                                   config.result_path + "/%d_%s_%.4f.pth" % (epoch + 1, config.model_type, fw_miou))
                    max_fwiou = fw_miou
                print("\n")
                print(miou)
                print("testing epoch loss: " + str(val_loss), "FWmIoU = %.4f" % fw_miou)
                writer.add_scalar('FWIoU/val', fw_miou, epoch + 1)
                writer.add_scalar('loss/val', val_loss, epoch + 1)
                for idx, name in enumerate(objects):
                    writer.add_scalar('iou/val' + name, miou[idx], epoch + 1)
            torch.cuda.empty_cache()
    writer.close()
    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=384)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-3)
    parser.add_argument('--model_type', type=str, default='HRNet_OCR', help='UNet/UNet++/RefineNet')
    parser.add_argument('--data_type', type=str, default='multi', help='single/multi')
    parser.add_argument('--loss', type=str, default='ce', help='ce/dice/mix')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam/adamw')
    parser.add_argument('--iscontinue', type=str, default=False, help='true/false')
    parser.add_argument('--smooth', type=str, default=False, help='true/false')

    parser.add_argument('--train_img_dir', type=str, default="../data/PCL/Stage2/train/image")
    parser.add_argument('--train_mask_dir', type=str, default="../data/PCL/Stage2/train/mask")
    parser.add_argument('--val_img_dir', type=str, default="../data/PCL/Stage2/val/image")
    parser.add_argument('--val_mask_dir', type=str, default="../data/PCL/Stage2/val/mask")
    parser.add_argument('--num_train', type=int, default=99016, help="4800/1600")
    parser.add_argument('--num_val', type=int, default=1016, help="1200/400")
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--result_path', type=str, default='./exp')

    config = parser.parse_args()
    train_val(config)
