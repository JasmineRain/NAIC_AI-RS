import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["MultiRendLoss"]


class MultiRendLoss(nn.Module):
    def __init__(self):
        super(MultiRendLoss, self).__init__()

    def forward(self, output, mask):
        coarse, stage1, stage2, stage3, stage4, stage5 = output.values()

        pred0 = F.interpolate(coarse, mask.shape[-2:], mode="bilinear", align_corners=True)

        rend1 = stage1[1]
        gt_points1 = sampling_features(mask, stage1[0], mode='bilinear', align_corners=True).argmax(dim=1)
        # print(rend1.shape, gt_points1.shape)
        point_loss1 = F.cross_entropy(rend1, gt_points1)

        rend2 = stage2[1]
        gt_points2 = sampling_features(mask, stage2[0], mode='bilinear', align_corners=True).argmax(dim=1)
        point_loss2 = F.cross_entropy(rend2, gt_points2)

        rend3 = stage3[1]
        gt_points3 = sampling_features(mask, stage3[0], mode='bilinear', align_corners=True).argmax(dim=1)
        point_loss3 = F.cross_entropy(rend3, gt_points3)

        rend4 = stage4[1]
        gt_points4 = sampling_features(mask, stage4[0], mode='bilinear', align_corners=True).argmax(dim=1)
        point_loss4 = F.cross_entropy(rend4, gt_points4)

        rend5 = stage5[1]
        gt_points5 = sampling_features(mask, stage5[0], mode='bilinear', align_corners=True).argmax(dim=1)
        point_loss5 = F.cross_entropy(rend5, gt_points5)

        mask = mask.argmax(dim=1)
        seg_loss = F.cross_entropy(pred0, mask)
        point_loss = point_loss1 + point_loss2 + point_loss3 + point_loss4 + point_loss5

        loss = seg_loss + point_loss

        return loss
