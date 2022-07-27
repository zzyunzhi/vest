import cv2
import numpy as np
from PIL import Image


def tensor2pilimage(image, width=None, height=None, minus1to1_normalized=False):
    r"""Convert a 3 dimensional torch tensor to a PIL image with the desired
    width and height.

    Args:
        image (3 x W1 x H1 tensor): Image tensor
        width (int): Desired width for the result PIL image.
        height (int): Desired height for the result PIL image.
        minus1to1_normalized (bool): True if the tensor values are in [-1,
        1]. Otherwise, we assume the values are in [0, 1].

    Returns:
        (PIL image): The resulting PIL image.
    """
    if len(image.size()) != 3:
        raise ValueError('Image tensor dimension does not equal = 3.')
    if image.size(0) != 3:
        raise ValueError('Image has more than 3 channels.')
    if minus1to1_normalized:
        # Normalize back to [0, 1]
        image = (image + 1) * 0.5
    image = image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    output_img = Image.fromarray(np.uint8(image))
    if width is not None and height is not None:
        output_img = output_img.resize((width, height), Image.BICUBIC)
    return output_img


def tensor2im(image_tensor, imtype=np.uint8, normalize=True,
              three_channel_output=True):
    r"""Convert tensor to image.

    Args:
        image_tensor (torch.tensor or list of torch.tensor): If tensor then
            (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.
        normalize (bool): Is the input image normalized or not?
            three_channel_output (bool): Should single channel images be made 3
            channel in output?

    Returns:
        (numpy.ndarray, list if case 1, 2 above).
    """
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x, imtype, normalize) for x in image_tensor]
    if image_tensor.dim() == 5 or image_tensor.dim() == 4:
        return [tensor2im(image_tensor[idx], imtype, normalize)
                for idx in range(image_tensor.size(0))]

    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(
                image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 and three_channel_output:
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        elif image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:, :, :3]
        return image_numpy.astype(imtype)


def tensor2flow(tensor, imtype=np.uint8):
    r"""Convert flow tensor to color image.

    Args:
        tensor (tensor) of
        If tensor then (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.

    Returns:
        (numpy.ndarray or normalized torch image).
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        tensor = [t for t in tensor if t is not None]
        if not tensor:
            return None
        return [tensor2flow(t, imtype) for t in tensor]
    if tensor.dim() == 5 or tensor.dim() == 4:
        return [tensor2flow(tensor[b]) for b in range(tensor.size(0))]

    tensor = tensor.detach().cpu().float().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=imtype)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
