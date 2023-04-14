from typing import Union, List, Callable, Optional, Dict
import urllib.parse
import os
import numpy as np
import torch
import torchvision
from .html_table import HTMLTableVisualizer
import logging

logger = logging.getLogger(__name__)
VCV_URL_PREFIX = "http://vcv.stanford.edu"
VCV_FILE_URL_PREFIX = f"{VCV_URL_PREFIX}/cgi-bin/file-explorer/?"


def setup_vi(path=None, exist_ok=True, title='default'):
    from ..common import vi_helper

    if path is None:
        from ..common import vi
        return vi, vi_helper
    if not exist_ok:
        assert not os.path.exists(path), (path, title)
    vi = HTMLTableVisualizer(path, title=title)
    vi.begin_html()
    vi_helper.print_url(vi)
    return vi, vi_helper


def tensor2im(image_tensor: Union[torch.Tensor, List[torch.Tensor], None]):
    """

    Args:
        image_tensor: (list of) shape [..., C, H, W]

    Returns:

    """
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x) for x in image_tensor]  # keep the list structure
    # if image_tensor.dim() == 5 or image_tensor.dim() == 4:
    #     return [tensor2im(image_tensor[idx])
    #             for idx in range(image_tensor.size(0))]

    # n_dim = len(image_tensor.shape)  # (..., c, h, w)
    image_tensor = image_tensor.movedim(-3, -1)

    image_numpy = image_tensor.cpu().float().numpy()
    # assume range (0, 1)
    image_numpy = image_numpy * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)  # (..., h, w, c)

    # to pil
    # output_img = Image.fromarray(np.uint8(image))
    # if width is not None and height is not None:
    #     output_img = output_img.resize((width, height), Image.BICUBIC)
    return image_numpy.astype(np.uint8)


def collect_tensor(tensor: torch.Tensor,
                   process_grid: Callable[[torch.Tensor], np.ndarray] = tensor2im,
                   pad_value=0, padding=0,
                   value_check: bool = True):
    """

    Args:
        tensor: expect range [0, 1]
        pad_value:
        padding:
        process_grid: e.g. tensor2im, tensor2flow
        value_check:

    Returns:

    """
    if value_check and (tensor.min() < -1e-3 or tensor.max() > 1 + 1e-3):
        print("checking min / max, not normalized? ")
        print(tensor.min(), tensor.max())
        # raise

    if len(tensor.shape) == 3:
        tensor = tensor[None]
    elif len(tensor.shape) == 4:  # (n, c, h, w)
        nrow_this = 1
    elif len(tensor.shape) == 5:  # (n, t, c, h, w)
        nrow_this = tensor.shape[1]

    tensor = tensor.flatten(end_dim=-4)
    if tensor.shape[0] == 1:
        # this is a hack since make_grid does not pad bs=1 tensors
        image_grid = torch.ones((tensor.shape[-3], tensor.shape[-2] + 2 * padding, tensor.shape[-1] + 2 * padding),
                                ) * pad_value
        image_grid.narrow(1, padding, tensor.shape[-2]).narrow(
            2, padding, tensor.shape[-1]
        ).copy_(tensor[0])

        if image_grid.shape[0] == 1:
            # output as rgb channels
            image_grid = image_grid.expand(3, image_grid.shape[1], image_grid.shape[2])

    else:
        image_grid = torchvision.utils.make_grid(tensor, nrow=nrow_this, pad_value=pad_value,
                                                 padding=padding)  # (b or b*d, c, h, w) -> (c, h, w)
        # make_grid will always output channel = 3
    image_grid_np = process_grid(image_grid)
    return image_grid_np


def print_viscam_url(path, verbose=True):
    path = os.path.abspath(path)
    # local file
    url = f"file:///{path}"
    if verbose:
        logger.info(f"[URL] {url}")
    return url


def print_vcv_url(path, verbose=True, nrow=4, query_kwargs: Optional[Dict] = None) -> str:
    path = os.path.abspath(path)

    filename = ''
    if os.path.isfile(path):
        path, filename = os.path.split(path)

    query = dict(
        dir=path,
        patterns_show=filename,
        patterns_highlight='',
        w=1500 // nrow, h=1500,
        n=nrow, autoplay=1, showmedia=1,
    )
    if query_kwargs is not None:
        query.update(query_kwargs)
    url_query = urllib.parse.urlencode(query)
    url = f'{VCV_FILE_URL_PREFIX}{url_query}'
    if verbose:
        logger.info(f"[URL] {url}")
    return url
