import torch.nn as nn
import torch
import torch.nn.functional as F
from vest.layers.conv import Conv2dBlock, Conv3dBlock


#https://github.com/google-research/google-research/blob/master/single_view_mpi/libs/nets.py

class UNet(nn.Module):
    def __init__(self, unet_cfg, num_input_channels):
        super().__init__()

        c_stack = []
        def down(c_active):
            c_stack.append(c_active)
            return c_active

        def up(c_active):
            return c_active + c_stack.pop()

        def conv(kernel, c, name):
            def f(c_active):
                self.__setattr__(name,
                                 Conv2dBlock(c_active, c,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             activation_norm_type=unet_cfg.activation_norm_type,
                                             weight_norm_type=unet_cfg.weight_norm_type,
                                             nonlinearity='relu', inplace_nonlinearity=True)
                                 )
                return c
            return f

        self.num_downsamples = unet_cfg.num_downsamples
        spec = self._get_spec(conv, up, down)

        c_active = num_input_channels
        for item in spec:
            c_active = item(c_active)

        self.out_channels = c_active

    def _get_spec(self, conv, up, down):
        num_downsamples = self.num_downsamples

        spec_down = [
            conv(7, 32, 'down1'),
            conv(7, 32, 'down1b'), down,
            conv(5, 64, 'down2'),
            conv(5, 64, 'down2b'), down,
            conv(3, 128, 'down3'),
            conv(3, 128, 'down3b'), down]

        assert num_downsamples >= 3 and num_downsamples <= 7

        if num_downsamples >= 4:
            spec_down = spec_down + [
            conv(3, 256, 'down4'),
            conv(3, 256, 'down4b'), down]
        if num_downsamples >= 5:
            spec_down = spec_down + [
                conv(3, 512, 'down5'),
                conv(3, 512, 'down5b'), down
            ]
        if num_downsamples >= 6:
            spec_down = spec_down + [
                conv(3, 512, 'down6'),
                conv(3, 512, 'down6b'), down,
            ]
        if num_downsamples >= 7:
            spec_down = spec_down + [
                conv(3, 512, 'down7'),
                conv(3, 512, 'down7b'), down,
            ]
        # conv(3, 512, 'down5'),
        # conv(3, 512, 'down5b'), down
        # conv(3, 512, 'down6'),
        # conv(3, 512, 'down6b'), down,
        # conv(3, 512, 'down7'),
        # conv(3, 512, 'down7b'), down,
        spec_mid = [conv(3, 512, 'mid1'),
                    conv(3, 512, 'mid2'), up, ]

        spec_up = []
        if num_downsamples >= 7:
            spec_up = spec_up + [
                conv(3, 512, 'up7'),
                conv(3, 512, 'up7b'), up,
            ]
        if num_downsamples >= 6:
            spec_up = spec_up + [
                conv(3, 512, 'up6'),
                conv(3, 512, 'up6b'), up,
            ]
        if num_downsamples >= 5:
            spec_up = spec_up + [
                conv(3, 512, 'up5'),
                conv(3, 512, 'up5b'), up,
            ]
        if num_downsamples >= 4:
            spec_up = spec_up + [
                conv(3, 256, 'up4'),
                conv(3, 256, 'up4b'), up,
            ]

        # conv(3, 512, 'up7'),
        # conv(3, 512, 'up7b'), up,
        # conv(3, 512, 'up6'),
        # conv(3, 512, 'up6b'), up,
        # conv(3, 512, 'up5'),
        # conv(3, 512, 'up5b'), up,
        spec_up = spec_up + [
                   conv(3, 128, 'up3'),
                   conv(3, 128, 'up3b'), up,
                   conv(3, 64, 'up2'),
                   conv(3, 64, 'up2b'), up,
                   conv(3, 64, 'post1'),
                   conv(3, 64, 'post2'),
                   conv(3, 64, 'up1'),
                   conv(3, 64, 'up1b')
                   ]

        spec = spec_down + spec_mid + spec_up

        return spec

    def forward(self, x):
        stack = []

        def down(t):
            stack.append(t)
            return F.max_pool2d(t, 2)

        def up(t):
            doubled = torch.repeat_interleave(torch.repeat_interleave(t, repeats=2, dim=-1), repeats=2, dim=-2)
            return torch.cat([doubled, stack.pop()], dim=-3)

        def conv(kernel, c, name):
            return self.__getattr__(name)

        spec = self._get_spec(conv, up, down)

        t = x
        for item in spec:
            t = item(t)

        return t
