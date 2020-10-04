import numpy as np
import torch
import torch.nn.functional as F


def is_near(point, mask, nearest_neighbor):
    nn = nearest_neighbor
    w, h = mask.shape[0], mask.shape[1]
    x, y = point
    mask = np.pad(mask, nn, 'edge')
    x += nn
    y += nn
    if w + nn > x and h + nn > y:
        x_i, y_i = int(x + 0.5), int(y + 0.5)
        # return True
        near = mask[x_i - nn:x_i + nn, y_i - nn:y_i + nn]
        if near.max() - near.min() != 0:
            if (x < w and y < h):
                return True
    return False


def get_edge_neighbor(img, k):
    w, h = img.shape
    padded = np.pad(img, k, 'edge')

    res = np.zeros(img.shape)

    for i in range(w):
        for j in range(h):
            neighbor = padded[i:i + 2 * k, j:j + 2 * k]
            _max = neighbor.max()
            _min = neighbor.min()
            res[i - k, j - k] = (_max - _min)

    return res


def is_near_v2(point, edge_k_neighbor):
    x, y = point
    x, y = int(x), int(y)
    return edge_k_neighbor[x][y] > 0


def sampling_points_v1(mask, N, k, beta, training=True, nearest_neighbor=2, new_if_near=True):
    w, h = mask.shape
    N = min(N, w * h)
    xy_min = [0, 0]
    xy_max = [w - 1, h - 1]
    points = np.random.uniform(low=xy_min, high=xy_max, size=(N, 2))
    print(points.shape)
    if (beta > 1 or beta < 0):
        print("beta should be in range [0,1]")
        return []

    # for the training, the mask is a hard mask
    if training:
        if beta == 0:
            return points
        res = []
        if new_if_near:
            edge_k_neighbor = get_edge_neighbor(mask, nearest_neighbor)
            for p in points:
                if is_near_v2(p, edge_k_neighbor):
                    res.append(p)
        else:
            for p in points:
                if is_near(p, mask, nearest_neighbor):
                    res.append(p)

        others = int((1 - beta) * k * w * h)
        not_edge_points = np.random.uniform(low=xy_min, high=xy_max, size=(others, 2))
        for p in not_edge_points:
            res.append(p)
        return res

    # for the inference, the mask is a soft mask
    if training == False:
        res = []
        for i in range(w):
            for j in range(h):
                if mask[i, j] > 0:
                    res.append((i, j))
        return res


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
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

        # uncertainty_map = torch.abs(mask.view(B, -1) - 0.5)
        # _, idx = uncertainty_map.topk(N, dim=1, largest=False)
        #
        # points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        # points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        # points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step

        # return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = sampling_features(mask, over_generation, align_corners=True)

    temp, _ = torch.sort(over_generation_map, dim=1, descending=True)
    temp = temp.detach()
    uncertainty_map = -1 * (temp[:, 0, :, :] - temp[:, 1, :, :])

    _, idx = uncertainty_map.topk(int(beta * N), dim=-1, largest=False)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.reshape(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)

