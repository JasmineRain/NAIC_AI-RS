import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["RendLoss"]


class RendLoss(nn.Module):
    def __init__(self):
        super(RendLoss, self).__init__()

    def forward(self, output, mask):

        pred = F.interpolate(output['coarse'], mask.shape[-2:], mode="bilinear", align_corners=True)
        gt_points = sampling_features(mask, output['points'], mode='bilinear', align_corners=True).argmax(dim=1)
        mask = mask.argmax(dim=1)
        seg_loss = F.cross_entropy(pred, mask)
        point_loss = F.cross_entropy(output['rend'], gt_points)

        loss = seg_loss + point_loss

        return loss
