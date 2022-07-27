import torch
import warnings
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from vest.third_party.midas.midas.transforms import Resize
# from vest.third_party.inpainting.MiDaS.monodepth_net import MonoDepthNet


DISP_RESCALE = 10
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero(as_tuple=False)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def compute_scale_and_shift_safe(prediction, target, mask):
    # FIXME: need stop gradient for scale and shift, same for trimmed product loss???
    scale, shift = compute_scale_and_shift(prediction, target, mask)

    m = scale <= 0
    if m.any():
        # set scale = 1
        # shift = - 1/M \sum_{i=1}^M (d_i - d_i^*)
        M = torch.sum(mask, (1, 2))  # (b,)
        new_scale = torch.ones_like(scale)
        new_shift = - 1/M * (prediction - target).sum((1, 2))

        scale[m] = new_scale
        shift[m] = new_shift
    return scale, shift


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        assert prediction.shape == target.shape == mask.shape, (prediction.shape, target.shape, mask.shape)
        if len(prediction.shape) == 4:  # (b, 1, h, w)
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)
            mask = mask.squeeze(1)
        else:
            assert len(prediction.shape) == 3, prediction.shape  # (b, h, w)

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # change inference code when scale < 0
        # if torch.any(scale < 0):
        #     print(scale)
        #     import ipdb; ipdb.set_trace()
        # scale, shift = compute_scale_and_shift_safe(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        data_loss = self.__data_loss(self.__prediction_ssi, target, mask)
        reg_loss = self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return data_loss, reg_loss, dict(scale=scale, shift=shift)

    # def __get_prediction_ssi(self):
    #     return self.__prediction_ssi
    #
    # prediction_ssi = property(__get_prediction_ssi)


# https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2


def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]

    return trimmed.sum() / (2 * M.sum())


def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


def get_median_scale(depth):
    # depth: (b, 1, h, w)
    # equation (5) from https://arxiv.org/pdf/1907.01341v3.pdf
    assert depth.shape[1] == 1 and len(depth.shape) == 4
    depth_flatten = depth.flatten(start_dim=1)  # (b, h * w)
    median = depth_flatten.median(dim=-1).values  # (b,)
    scale = (depth_flatten - median.view(-1, 1)).abs().mean(-1)  # (b,)
    assert median.shape == scale.shape
    return median, scale


class L1MatchMedianScaleLoss(nn.Module):
    def forward(self, prediction, target):
        median_pred, scale_pred = get_median_scale(prediction)
        median_targ, scale_targ = get_median_scale(target)

        centered_pred = (prediction - median_pred.view(-1, 1, 1, 1)) / scale_pred.view(-1, 1, 1, 1)
        centered_targ = (target - median_targ.view(-1, 1, 1, 1)) / scale_targ.view(-1, 1, 1, 1)

        extra_info = dict(
            median_prediction=median_pred,
            scale_prediction=scale_pred,
            centered_prediction=centered_pred,
            median_target=median_targ,
            scale_target=scale_targ,
            centered_target=centered_targ,
        )

        return F.l1_loss(input=centered_pred, target=centered_targ), extra_info


class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        print('using trimmed procrusted loss', alpha, scales, reduction)
        super(TrimmedProcrustesLoss, self).__init__()

        # self.__data_loss = TrimmedMAELoss(reduction=reduction)
        self.__data_loss = trimmed_mae_loss
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        if len(prediction.shape) == 4:  # (b, 1, h, w)
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)
            mask = mask.squeeze(1)
        else:
            assert len(prediction.shape) == 3, prediction.shape  # (b, h, w)

        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        # print(prediction.max(), prediction.min(), self.__prediction_ssi.max(), self.__prediction_ssi.min())
        # print(target.max(), target.min(), target_.max(), target_.min())
        data_loss = self.__data_loss(self.__prediction_ssi, target_, mask)
        # if self.__alpha > 0:
        reg_loss = self.__alpha * self.__regularization_loss(
            self.__prediction_ssi, target_, mask
        )
        return data_loss, reg_loss


def normalize_disp(disp):
    # normalize to [0, 1]
    disp = disp - disp.min()
    if disp.max() < 1e-5:
        print('blank disp')
        return None
    disp = disp / disp.max()
    return disp


class NormalizedDispMSELoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        print('using normalized disp mse loss', alpha, scales, reduction)
        super().__init__()

        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

    def forward(self, prediction, target, mask):
        assert prediction.shape == target.shape == mask.shape, (prediction.shape, target.shape, mask.shape)
        if len(prediction.shape) == 4:  # (b, 1, h, w)
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)
            mask = mask.squeeze(1)
        else:
            assert len(prediction.shape) == 3, prediction.shape  # (b, h, w)

        prediction = normalize_disp(prediction)  # FIXME: don't adjust prediction?
        target_ = normalize_disp(target)
        if target_ is None:  # invalid target
            return torch.zeros(()).cuda(), torch.zeros(()).cuda()
        target = target_

        data_loss = F.mse_loss(prediction, target)
        reg_loss = self.__alpha * self.__regularization_loss(prediction, target, mask)

        return data_loss, reg_loss


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class RelativeDepthLoss(nn.Module):
    def forward(self, prediction, target, n_samples):
        # sample coordinates
        bs, _, h, w = prediction.shape
        x0 = np.random.randint(h, size=(bs, n_samples))
        y0 = np.random.randint(w, size=(bs, n_samples))
        for i in range(prediction.shape[0]):
            p0 = prediction[i][x0[i], y0[i]]
            p1 = prediction[i][x1[i], y1[i]]

            p0_star = target[i][x0[i], y0[i]]
            p1_star = None
            ratio = p0_star.div(p1_star)
            l = torch.zeros_like(ratio)
            l[ratio.geq(1 + tau)] = 1
            l[ratio.leq(1 / (1 + tau))] = -1

            phi = torch.zeros_like(l)
            phi[l != 0] = torch.log(1 + torch.exp(-l * (p0 - p1)))
            phi[l == 0] = (p0 - p1).pow(2)


# https://github.com/yifjiang/relative-depth-using-pytorch/blob/master/models/criterion/relative_depth.py
# https://github.com/princeton-vl/relative_depth/blob/master/src/experiment/criterion/relative_depth.lua
class RelativeDepthLoss(nn.Module):

    def __loss_func_arr(self, z_A, z_B, ground_truth):
        mask = torch.abs(ground_truth)
        z_A = z_A[0]
        a_B = z_B[0]
        return mask*torch.log(1+torch.exp(-ground_truth*(z_A-z_B)))+(1-mask)*(z_A-z_B)*(z_A-z_B)

    def forward(self, input, target):
        self.input = input
        self.target = target
        self.output = torch.zeros(()).cuda()
        n_point_total = 0
        cpu_input = input
        for batch_idx in range(0,cpu_input.size()[0]):
            n_point_total+=target[batch_idx]['n_point']

            x_A_arr = target[batch_idx]['x_A']
            y_A_arr = target[batch_idx]['y_A']
            x_B_arr = target[batch_idx]['x_B']
            y_B_arr = target[batch_idx]['y_B']

            batch_input = cpu_input[batch_idx, 0]
            z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
            z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())

            ground_truth_arr = target[batch_idx]['ordianl_relation']
            self.output += torch.sum(self.__loss_func_arr(z_A_arr, z_B_arr, ground_truth_arr))

        return self.output/n_point_total


# https://github.com/A-Jacobson/Depth_in_The_Wild/blob/master/criterion.py
class RelativeDepthLoss(nn.Module):
    def __init__(self):
        super(RelativeDepthLoss, self).__init__()

    def ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        pred_depth = z_A - z_B
        log_loss = torch.mean(torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        squared_loss = torch.mean(pred_depth[target == 0] ** 2)  # if pred depth is not zero adds to loss
        return log_loss + squared_loss

    def forward(self, output, target):
        total_loss = 0
        for index in range(len(output)):
            x_A = target['x_A'][index].long()
            y_A = target['y_A'][index].long()
            x_B = target['x_B'][index].long()
            y_B = target['y_B'][index].long()

            z_A = output[index][0][x_A, y_A]  # all "A" points
            z_B = output[index][0][x_B, y_B]  # all "B" points

            total_loss += self.ranking_loss(z_A, z_B, target['ordinal_relation'][index])

        return total_loss / len(output)



def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}".format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))



def spatial_gradient(input: torch.Tensor, order: int = 1, normalized: bool = True) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    .. image:: _static/img/spatial_gradient.png

    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))
    # allocate kernel
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.transpose(0, 1)
    kernel = torch.stack([kernel_x, kernel_y])

    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 2
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)



def sobel(input: torch.Tensor, normalized: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Computes the Sobel operator and returns the magnitude per channel.

    .. image:: _static/img/sobel.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        normalized: if True, L1 norm of the kernel is set to 1.
        eps: regularization number to avoid NaN during backprop.

    Return:
        the sobel edge gradient magnitudes map with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_edges.html>`__.

    Example:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = sobel(input)  # 1x3x4x4
        >>> output.shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

    # comput the x/y gradients
    edges: torch.Tensor = spatial_gradient(input, normalized=normalized)

    # unpack the edges
    gx: torch.Tensor = edges[:, :, 0]
    gy: torch.Tensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)

    return magnitude