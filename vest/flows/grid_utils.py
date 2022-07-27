from vest.model_utils.fs_vid2vid import get_grid
import torch
import torch.nn.functional as F


def make_coordinate_grid(spatial_size, type):
    raise NotImplementedError

    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def grid_to_norm_flow(grid):
    # coordinates to flow
    bs, h, w, _ = grid.shape  # (b, h, w, 2)
    residual = get_grid(batchsize=bs, size=(h, w)).to(grid)
    norm_flow = grid.permute(0, 3, 1, 2) - residual
    return norm_flow


def grid_to_flow(grid):
    bs, h, w, _ = grid.shape  # (b, h, w, 2)
    norm_flow = grid_to_norm_flow(grid)
    flow = torch.cat([norm_flow[:, 0:1, :, :] * ((w - 1.0) / 2.0),
                      norm_flow[:, 1:2, :, :] * ((h - 1.0) / 2.0)], dim=1)

    return flow


def resample(image, flow, padding_mode='border'):
    r"""Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor) : Image to resample.
        flow (Nx2xHxW tensor) : Optical flow to resample the image.
    Returns:
        output (NxCxHxW tensor) : Resampled image.
    """
    output = F.grid_sample(image, flow_to_grid(flow), mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)
    return output


def flow_to_norm_flow(flow):
    assert flow.shape[1] == 2
    _, _, h, w = flow.shape
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    return flow


def flow_to_grid(flow):
    assert flow.shape[1] == 2
    b, _, h, w = flow.shape
    grid = get_grid(b, (h, w)).to(flow)
    flow = flow_to_norm_flow(flow)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    return final_grid


def compute_flow_conf(flow_backward, im1, im2, threshold=0.02):
    # expect pixel range [-1, 1] for the default threshold
    diff = im1 - resample(im2, flow_backward)
    norm = diff.pow(2).sum(dim=1, keepdim=True)
    conf = (norm < threshold).float()
    return conf


def create_affine_flow(img_size, driving_region_params, source_region_params):
    h, w = img_size
    bs, n_plane, _ = driving_region_params['shift'].shape
    identity_grid = make_coordinate_grid((h, w), type=source_region_params['shift'].type())
    identity_grid = identity_grid.view(1, 1, h, w, 2)
    coordinate_grid = identity_grid - driving_region_params['shift'].view(bs, n_plane, 1, 1, 2)
    affine = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))
    affine = affine.unsqueeze(-3).unsqueeze(-3)
    affine = affine.repeat(1, 1, h, w, 1, 1)
    coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
    coordinate_grid = coordinate_grid.squeeze(-1)

    driving_to_source = coordinate_grid + source_region_params['shift'].view(bs, n_plane, 1, 1, 2)

    return driving_to_source


def create_affine_flow_v2(img_size, shift, affine):
    """

    Args:
        image:
        shift: (b, d, 2)
        affine: (b, d, 4)

    Returns:

    """
    h, w = img_size
    bs, n_plane, _ = shift.shape
    identity_grid = make_coordinate_grid((h, w), type=shift.type())
    identity_grid = identity_grid.view(1, 1, h, w, 2)
    coordinate_grid = identity_grid

    affine = affine.unsqueeze(-3).unsqueeze(-3)
    affine = affine.repeat(1, 1, h, w, 1, 1)
    coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
    coordinate_grid = coordinate_grid.squeeze(-1)

    id_to_source = coordinate_grid + shift.view(bs, n_plane, 1, 1, 2)

    # id_to_source = id_to_source.movedim(-1, -3)

    return id_to_source


def warp(colors, coordinates):
    """

    Args:
        colors: (b, d, c, h, w) or (b, c, h, w)
        coordinates: (b, d, 2, h, w) or (b, c, h, w) (correspondingly)

    Returns:

    """
    warped = F.grid_sample(colors.flatten(end_dim=-4),
                           coordinates.flatten(end_dim=-4),
                           align_corners=False,
                           mode='bilinear',
                           padding_mode='zeros')  # or borders?
    warped = warped.view(*colors.shape)
    return warped
