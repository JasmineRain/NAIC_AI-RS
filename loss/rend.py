import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["RendLoss"]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        N = target.size(0)
        smooth = 1

        input_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class RendLoss(nn.Module):
    def __init__(self):
        super(RendLoss, self).__init__()

    def forward(self, output, mask):

        pred = F.interpolate(output['coarse'], mask.shape[-2:], mode="bilinear", align_corners=True)
        gt_points = sampling_features(mask, output['points'], mode='bilinear', align_corners=True).argmax(dim=1)
        mask = mask.argmax(dim=1)
        seg_loss = F.cross_entropy(pred, mask)
        # print(pred.shape, mask.shape)
        # print(output['rend'].squeeze().shape, gt_points.shape)
        point_loss = F.cross_entropy(output['rend'], gt_points)

        loss = seg_loss + point_loss

        return loss
