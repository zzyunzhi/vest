from vest.utils import mpi
import cv2
import urllib.parse
import os
import numpy as np
import imageio
import torch
import torchvision
from vest.utils.visualization.common import tensor2flow


def get_denormalize_rgb(cfg):
    if cfg.data.input_types[0].images.normalize:  # color in (-1, 1), alpha in (0, 1)
        # input rgb range (-1, 1)
        # output rgb range (0, 1)
        return lambda t: (t + 1) * 0.5

    # input, output rgb range (0, 1)
    return lambda t: t


def my_tensor2im(image_tensor):
    # different from vest.utils.visualization!!
    # their default is normalize = True

    r"""Convert tensor to image.

    Args:
        image_tensor: Tensor of shape [..., C, H, W]
        imtype (np.dtype): Type of output image.
        normalize (bool): Is the input image normalized or not?
            three_channel_output (bool): Should single channel images be made 3
            channel in output?

    Returns:
        (numpy.ndarray, list if case 1, 2 above).
    """

    # n_dim = len(image_tensor.shape)  # (..., c, h, w)
    image_tensor = image_tensor.movedim(-3, -1)
    # image_tensor = image_tensor.permute(*list(np.arange(n_dim-3)), n_dim-2, n_dim-1, n_dim-3)

    image_numpy = image_tensor.cpu().float().numpy()
    # assume range (0, 1)
    image_numpy = image_numpy * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(np.uint8)


def my_tensor2flow(tensor, imtype=np.uint8, ignore_mag=False, ignore_ang=False):
    r"""Convert flow tensor to color image.

    Args:
        tensor (tensor) of
        If tensor then (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.

    Returns:
        (numpy.ndarray or normalized torch image).
    """
    tensor = tensor.detach().cpu().float().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.ones((tensor.shape[0], tensor.shape[1], 3), dtype=imtype) * 255
    # hsv[..., 0] = 0
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    if not ignore_ang:
        hsv[..., 0] = ang * 180 / np.pi / 2
    if not ignore_mag:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def collect_tensor(tensor, normalize_fn=lambda t: t, pad_value=0, padding=2, process_grid=None):
    assert tensor.shape[-3] in [1, 2, 3, 4], tensor.shape
    tensor = normalize_fn(tensor)
    if len(tensor.shape) == 3:
        tensor = tensor[None]
    
    if tensor.shape[-3] != 2 and (tensor.min() < -1e-3 or tensor.max() > 1+1e-3):
        # tensor.shape[-3] == 2 -> flow -> unnormalized?
        print()
        print("inproper min / max, not normalized? ")
        print(tensor.min(), tensor.max())
        print()
        print()
        # raise
    # each row = one sample in the batch
    # each column = one mpi plane
    nrow_this = 1 if len(tensor.shape) == 4 else tensor.shape[1]
    assert len(tensor.shape) in [4, 5], tensor.shape
    tensor = tensor.flatten(end_dim=-4)
    if tensor.shape[0] == 1:
        # this is a hack since make_grid does not pad bs=1 tensors
        image_grid = torch.ones((tensor.shape[-3], tensor.shape[-2] + 2 * padding, tensor.shape[-1] + 2 * padding),
                                ) * pad_value
        image_grid.narrow(1, padding, tensor.shape[-2]).narrow(
            2, padding, tensor.shape[-1]
        ).copy_(tensor[0])

    else:
        image_grid = torchvision.utils.make_grid(tensor, nrow=nrow_this, pad_value=pad_value, padding=padding)  # (b or b*d, c, h, w) -> (c, h, w)
    process_grid = process_grid or (tensor2flow if image_grid.shape[-3] == 2 else my_tensor2im)
    image_grid_np = process_grid(image_grid)
    return image_grid_np

def normalize(t, amin, amax):
    assert amin < amax, (amin, amax)
    return (t - amin) / max(amax - amin, 1e-3)

def visualize_rgba(rgba, save_dir=None, save_prefix=None):
    # assert rgba has shape (b, d, c+1, h, w)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    name_to_images = dict()

    def save(image, name):
        name_to_images[name] = image
        if save_dir is not None:
            imageio.imwrite(os.path.join(save_dir, f"{save_prefix}_{name}.png"), image)

    color = rgba[..., :-1, :, :]
    alpha = rgba[..., -1:, :, :]

    transmittance = mpi.layer_weights(alpha)
    # refined_color = color * transmittance

    save(collect_tensor(rgba), 'rgba_layers')
    save(collect_tensor(torch.cat([color, transmittance], dim=-3)), 'refined_rgba_layers')

    save(collect_tensor(color), 'color_layers')
    # save(collect_tensor(refined_color), 'refined_color_layers')

    save(collect_tensor(torch.cat([torch.zeros_like(color), alpha], dim=-3)), 'alpha_layers')
    save(collect_tensor(torch.cat([torch.zeros_like(color), transmittance], dim=-3)), 'transmittance_layers')

    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, 'composed'), exist_ok=True)

    def save_composed(image, name):
        name_to_images[name] = image
        if save_dir is not None:
            imageio.imwrite(os.path.join(save_dir, 'composed', f"{save_prefix}_{name}.png"), image)

    save_composed(collect_tensor((color * transmittance).sum(-4)), 'composed')

    return name_to_images


def visualize_rgb(rgb, normalize_rgb_fn, save_dir, save_prefix):
    def save(image, name):
        imageio.imwrite(os.path.join(save_dir, f"{save_prefix}_{name}.png"), image)
    save(collect_tensor(rgb, normalize_rgb_fn), 'color_layers')


def new_hstack_layouts(*image_list, pad_value=242):
    max_height = max(image.shape[0] for image in image_list)  # (h, w, c)

    # pad all images to be the same height

    def pad(image):
        new_image = np.ones((max_height, image.shape[1], image.shape[2]), dtype=image.dtype) * pad_value
        start_i = (max_height - image.shape[0]) // 2
        new_image[start_i:start_i+image.shape[0], :, :] = image
        return new_image

    image_list = [pad(image) for image in image_list]

    delim = np.ones_like(image_list[0]) * pad_value
    delim = delim[:, :max(delim.shape[1]//40 * 4, 4), :]
    new_image_list = [image_list[0]]
    for i in range(1, len(image_list)):
        new_image_list.append(delim)
        new_image_list.append(image_list[i])
    final_image = np.hstack(new_image_list)
    return final_image


def stack_layouts(layouts, pad_value=255):
    assert isinstance(layouts, list)
    assert isinstance(layouts[0], list)
    assert isinstance(layouts[0][0], np.ndarray)

    # assume all images in layouts have the same shape (h, w, c)
    return vstack_layouts(
        *[new_hstack_layouts(*row, pad_value=pad_value) for row in layouts],
        pad_value=pad_value,
    )


def vstack_layouts(*image_list, pad_value):
    delim = np.ones_like(image_list[0]) * pad_value  # (h, w, c)
    delim = delim[:max(delim.shape[0]//40 * 4, 4), :, :]
    new_image_list = [image_list[0]]
    for i in range(1, len(image_list)):
        new_image_list.append(delim)
        new_image_list.append(image_list[i])
    final_image = np.vstack(new_image_list)
    return final_image


def hstack_layouts(*frames_list):
    multi_frames = True
    if not isinstance(frames_list[0], list):
        multi_frames = False
        assert all(isinstance(f, np.ndarray) for f in frames_list)
        # each element in frames_list is an np array (one single frame)
        frames_list = [[f] for f in frames_list]

    assert all([len(frames) == len(frames_list[0]) for frames in frames_list]), [len(frames) for frames in frames_list]
    n_frames = len(frames_list[0])

    # rgb -> rgba
    for frames in frames_list:
        # if frames[0].shape[-1] == 3:
        #     for i in range(n_frames):
        #         # mutable
        #         frame_rgba = np.concatenate([frames[i], np.ones((*frames[i].shape[:-1], 1), dtype=frames[i].dtype) * 255], axis=-1)
        #         frames[i] = frame_rgba
        assert frames[0].shape[-1] in [1, 3, 4], frames[0].shape

    # insert deliminators
    delim = np.zeros_like(frames_list[0][0])  # (h, w, c)
    delim = delim[:, :max(delim.shape[1]//40 * 4, 4), :]
    delim_frames = [delim for _ in range(n_frames)]
    new_frames_list = [frames_list[0]]
    for i in range(1, len(frames_list)):
        new_frames_list.append(delim_frames)
        new_frames_list.append(frames_list[i])

    final_frames = [np.hstack([frames[t] for frames in new_frames_list]) for t in range(n_frames)]
    if not multi_frames:
        return final_frames[0]
    return final_frames


def print_url(logdir, filename_prefix=None, filenames='', ext='.png', **kwargs):
    if not isinstance(filenames, list):
        filenames = [filenames]

    def get_query(filename, *, nrow=1, w=None, h=None):
        print(filename)

        return dict(
            dir=os.path.abspath(logdir),
            patterns_show=f"*{filename}*{ext}",
            patterns_highlight='',
            w=w or (1500//nrow), h=h or 1500,
            n=nrow, autoplay=1, showmedia=1
        )

    url_base = 'http://viscam2.stanford.edu:8080/cgi-bin/file-explorer/?'

    for filename in filenames:
        query = get_query(filename, **kwargs)

    url_query = urllib.parse.urlencode(query)
    print(url_base + url_query)
    print()
