import torch.nn as nn
import lpips
import torch.nn.functional as F
import torch

""" 
All functions expect input in range [0, 1]
"""

# class PNSR(nn.Module):
#     @staticmethod
#     def forward(input, target, max_val=1):
#         return 10.0 * torch.log10(max_val ** 2 / F.mse_loss(input, target, reduction='mean'))


def apply_crop(x):
    # crop 5%
    splat_bdry_ignore = 0.05

    *_, loss_h, loss_w = x.shape

    x_min = int(round(loss_w * splat_bdry_ignore))
    x_max = loss_w - x_min
    y_min = int(round(loss_h * splat_bdry_ignore))
    y_max = loss_h - y_min
    # centre_mask = tf.ones((batch_size, y_max - y_min, x_max - x_min))
    # padding = tf.constant([[0, 0], [y_min, loss_h - y_max],
    #                        [x_min, loss_w - x_max]])
    # centre_mask = tf.pad(centre_mask, padding)
    centre_mask = torch.ones((y_max - y_min, x_max - x_min)).to(x)
    padding = (x_min, loss_w - x_max, y_min, loss_h - y_max)
    centre_mask = F.pad(centre_mask, padding)

    return x * centre_mask

#
# class LDIPwiseSplatLoss(nn.Module):
#     def __init__(self, crop):
#         super().__init__()
#         self.crop = crop
#
#
#     def forward(self, input, target):
#         loss = F.l1_loss(input=input, target=target, reduction='none').mean(-3)
#         if self.crop:
#             loss = self.apply_crop(loss)
#
#         return loss.sum()

# https://github.com/facebookresearch/synsin/blob/501ec49b11030a41207e7b923b949fab8fd6e1b5/evaluation/metrics.py
from vest.third_party.synsin.models.losses.ssim import ssim
# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)

# The PSNR metric
def psnr(img1, img2, mask=None):
    b = img1.size(0)
    if not (mask is None):
        b = img1.size(0)
        mse_err = (img1 - img2).pow(2) * mask
        mse_err = mse_err.view(b, -1).sum(dim=1) / (
            3 * mask.view(b, -1).sum(dim=1).clamp(min=1)
        )
    else:
        mse_err = (img1 - img2).pow(2).view(b, -1).mean(dim=1)

    psnr = 10 * (1 / mse_err).log10()
    return psnr


class PNSR(nn.Module):
    @staticmethod
    def forward(input, target):
        return psnr(input, target)

class SSIMMetric(nn.Module):
    @staticmethod
    def forward(input, target):
        return ssim_metric(input, target)


class LPIPSVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = lpips.LPIPS(net='vgg')

    def forward(self, input, target):
        input = input * 2 - 1
        target = target * 2 - 1
        return self.m(input, target)


class LPIPSAlex(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = lpips.LPIPS(net='alex')

    def forward(self, input, target):
        input = input * 2 - 1
        target = target * 2 - 1
        return self.m(input, target)


class LPIPSVGGMINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = lpips.LPIPS(net='vgg')

    def forward(self, input, target):
        return self.m(input, target)


class LPIPSAlexMINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = lpips.LPIPS(net='alex')

    def forward(self, input, target):
        return self.m(input, target)


class LPIPSVGGVideoAutoencoder(nn.Module):
    def __init__(self):
        # https://www.vangoghmuseum.nl/en/collection/s0047v1962
        super().__init__()
        from vest.third_party.videoautoencoder.util.pretrained_networks import PNet
        vgg16 = PNet(use_gpu=True)
        vgg16.eval()
        self.m = vgg16

    def forward(self, input, target):
        return self.m(input * 2 - 1, target * 2 - 1).clamp_(max=10000)
