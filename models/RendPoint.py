import numpy as np
import torch
import torch.nn.functional as F


def sampling_features(feature, points, **kwargs):
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(feature, 2.0 * points - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def sampling_points_v2(mask, N=4, k=3, beta=0.75, training=True):
    assert mask.dim() == 4

    device = mask.device
    B, C, H, W = mask.shape

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)

        temp, _ = torch.sort(mask, dim=1, descending=True)
        temp = temp.detach()
        uncertainty_map = -1 * (temp[:, 0, :, :] - temp[:, 1, :, :])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1, largest=True)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = sampling_features(mask, over_generation, align_corners=True)

    temp, _ = torch.sort(over_generation_map, dim=1, descending=True)
    temp = temp.detach()
    uncertainty_map = -1 * (temp[:, 0, :] - temp[:, 1, :])

    _, idx = uncertainty_map.topk(int(beta * N), dim=-1, largest=True)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.reshape(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)

