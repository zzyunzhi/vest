import torch
import torch.nn.functional as F


def resample(image, flow):
    r"""Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor) : Image to resample.
        flow (Nx2xHxW tensor) : Optical flow to resample the image.
    Returns:
        output (NxCxHxW tensor) : Resampled image.
    """
    assert flow.shape[1] == 2
    b, c, h, w = image.size()
    grid = get_grid(b, (h, w)).to(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    try:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
    except Exception:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border')
    return output

def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    r"""Get a grid ranging [-1, 1] of 2D/3D coordinates.

    Args:
        batchsize (int) : Batch size.
        size (tuple) : (height, width) or (depth, height, width).
        minval (float) : minimum value in returned grid.
        maxval (float) : maximum value in returned grid.
    Returns:
        t_grid (4D tensor) : Grid of coordinates.
    """
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError('Dimension can only be 2 or 3.')
    x = torch.linspace(minval, maxval, cols)
    x = x.view(1, 1, 1, cols)
    x = x.expand(batchsize, 1, rows, cols)

    y = torch.linspace(minval, maxval, rows)
    y = y.view(1, 1, rows, 1)
    y = y.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([x, y], dim=1)

    if len(size) == 3:
        z = torch.linspace(minval, maxval, deps)
        z = z.view(1, 1, deps, 1, 1)
        z = z.expand(batchsize, 1, deps, rows, cols)

        t_grid = t_grid.unsqueeze(2).expand(batchsize, 2, deps, rows, cols)
        t_grid = torch.cat([t_grid, z], dim=1)

    t_grid.requires_grad = False
    return t_grid
def concat_frames(prev, now, n_frames):
    r"""Concat previous and current frames and only keep the latest $(n_frames).
    If concatenated frames are longer than $(n_frames), drop the oldest one.

    Args:
        prev (NxTxCxHxW tensor): Tensor for previous frames.
        now (NxCxHxW tensor): Tensor for current frame.
        n_frames (int): Max number of frames to store.
    Returns:
        result (NxTxCxHxW tensor): Updated tensor.
    """
    now = now.unsqueeze(1)
    if prev is None:
        return now
    if prev.shape[1] == n_frames:
        prev = prev[:, 1:]
    return torch.cat([prev, now], dim=1)
