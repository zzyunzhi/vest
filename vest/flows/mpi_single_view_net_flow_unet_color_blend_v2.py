import vest.utils.mpi as mpi
from vest.embedders.mpi_net_single_view import UNet as SingleViewUNet
from vest.flows.grid_utils import grid_to_flow
from tu.ddp import master_only_print
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from vest.layers import Conv2dBlock


def get_base_conv_block(flow_cfg):
    base_conv_block = partial(Conv2dBlock,
                              kernel_size=flow_cfg.kernel_size,
                              padding=flow_cfg.kernel_size // 2,
                              weight_norm_type=flow_cfg.weight_norm_type,
                              activation_norm_type=flow_cfg.activation_norm_type,
                              nonlinearity='leakyrelu')

    return base_conv_block


def get_paired_input_image_channel_number(data_cfg):
    num_channels = 0
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_image:
                num_channels += data_type[k].num_channels
    return num_channels


class BaseMPI(nn.Module):
    def __init__(self, flow_cfg, data_cfg):
        super().__init__()
        self.num_frames = data_cfg.num_frames_G
        self.n_plane = flow_cfg.n_plane
        self.n_channels = get_paired_input_image_channel_number(data_cfg)
        self.img_size = data_cfg.img_size

        if hasattr(flow_cfg, 'flow_dof'):
            # backward compatible
            if flow_cfg.flow_dof == -1:
                self.flow_dof = 2
                self.dense_flow = True
            else:
                self.flow_dof = flow_cfg.flow_dof
                self.dense_flow = False

        if data_cfg.input_types[0].images.normalize:
            # need (-1, 1)
            denormalize_rgb = lambda t: t
            self.rgb_range = (-1, 1)
        else:
            # need (0, 1)
            denormalize_rgb = lambda t: (t+1) * 0.5
            self.rgb_range = (0, 1)
        self._denormalize_rgb = denormalize_rgb  # bad naming

    # the rest are helper methods

    def _register_alpha_bias(self):
        # register alpha pre-tanh bias
        alpha = 1.0 / torch.arange(1, self.n_plane + 1)
        alpha = alpha.view(self.n_plane, 1, 1, 1).expand(self.n_plane, 1, *self.img_size)  # (d, 1, h, w)
        alpha = torch.atanh(2.0 * alpha - 1.0)
        alpha = alpha[1:]  # (d-1, 1, h, w) # no bias for background
        self.register_buffer('alpha_bias', alpha)

    def _register_theta_bias(self):
        theta_bias = torch.tensor([[1, 0, 0], [0, 1, 0]])
        theta_bias = theta_bias.view(1, 2, 3).expand(self.n_plane, 2, 3)  # (d, 2, 3)
        self.register_buffer('theta_bias', theta_bias)

    def forward(self, prev_images, gt_features=None):
        raise NotImplementedError

    def _activate_alpha(self, conv_out):
        alpha = conv_out
        alpha = (torch.tanh(alpha + self.alpha_bias) + 1) * 0.5

        alpha = torch.cat([torch.ones((alpha.shape[0], 1, 1, *self.img_size), device=alpha.device),
                           alpha], 1)
        # (b, d, 1, h, w)
        return alpha

    @staticmethod
    def apply_warp_op(warp_in_layers, mpi_features):
        # TODO: do grid_sample on cpu

        # assert os.environ['ALIGN_CORNERS'] in ['0', '1'], os.environ['ALIGN_CORNERS']
        warp_out_layers = F.grid_sample(input=warp_in_layers.flatten(0, 1),
                                        grid=mpi_features['coordinates'].flatten(0, 1),
                                        mode='bilinear',
                                        padding_mode='border',
                                        # align_corners=os.environ['ALIGN_CORNERS'] == '1',
                                        align_corners=True,
                                        ).unflatten(0, warp_in_layers.shape[:2])

        return warp_out_layers


class MPIEmbedder(BaseMPI):
    def __init__(self, flow_cfg, data_cfg):
        super().__init__(flow_cfg, data_cfg)

        """ MPI networks """

        channels_in = self.n_channels * (self.num_frames - 1)
        assert not flow_cfg.input_disparity
        assert not flow_cfg.input_flow
        assert not flow_cfg.input_norm_flow
        assert not flow_cfg.input_eulerian
        net = SingleViewUNet(flow_cfg, channels_in)
        self.mpi_net = net

        """ plane expansion networks """

        assert not flow_cfg.plane_input_disparity
        assert not flow_cfg.plane_input_flow
        assert not flow_cfg.plane_input_norm_flow
        assert not flow_cfg.plane_input_eulerian
        channels_in = net.out_channels
        base_conv_block = get_base_conv_block(flow_cfg)
        self.plane_expansion = base_conv_block(channels_in, self.n_plane * net.out_channels)
        self.plane_dim = net.out_channels

        """ predict alpha, theta layers """

        assert flow_cfg.use_affine_motion
        self.final_net = nn.Sequential(
            base_conv_block(net.out_channels, net.out_channels),
            base_conv_block(net.out_channels, 1 + 6,
                            kernel_size=7, padding=3, nonlinearity='none'),
        )
        self._register_alpha_bias()
        self._register_theta_bias()

        self.normalize_theta_weight = flow_cfg.normalize_theta_weight
        self.theta_output_multiplier = flow_cfg.theta_output_multiplier

        """ predict color layers """

        self.output_no_color = flow_cfg.output_no_color  # disable color prediction
        self.output_residual_color = flow_cfg.output_residual_color
        if self.output_no_color:
            master_only_print('disable color prediction')
        if self.output_residual_color:
            master_only_print('output residual color')
        if not self.output_no_color:
            self.bg_net = nn.Conv2d(net.out_channels, self.n_channels, kernel_size=(3, 3), padding=1)

        self.affine_grid_on_cpu = flow_cfg.affine_grid_on_cpu
        self.grid_sample_on_cpu = flow_cfg.grid_sample_on_cpu
        self.front_depth = flow_cfg.front_depth
        self.back_depth = flow_cfg.back_depth

        assert not flow_cfg.use_gradient_checkpoint

    def forward(self, prev_images, gt_features=None):
        mpi_features = dict()

        bs = prev_images.shape[0]

        """ MPI prediction """

        net_in = prev_images.flatten(1, 2)
        out = self.mpi_net(net_in)  # (b, t, c, h, w) -> (b', c, h, w)

        if not self.output_no_color:
            bg = self.bg_net(out)  # (b, c, h, w)

        net_in = out
        out = self.plane_expansion(net_in)
        out = out.unflatten(-3, (self.n_plane, self.plane_dim))  # (b, d, ?, h, w)

        batch_d = lambda t: t.flatten(0, 1)
        unbatch_d = lambda t: t.unflatten(0, (bs, self.n_plane))

        out = self.final_net(batch_d(out))
        out = unbatch_d(out)

        alpha, theta = out.split((1, 6), dim=-3)

        # activate alpha
        alpha = alpha[:, 1:]  # ignore background alpha layer, to be filled with 1
        alpha_layers = self._activate_alpha(alpha)  # (b', d, 1, h, w)

        if not self.output_no_color:
            # activate color
            bg = self._denormalize_rgb(torch.tanh(bg))
            fg = prev_images[:, -1]  # (b', c, h, w)
            if self.output_residual_color:
                bg = bg + fg
            blend = mpi.layer_visibility(alpha_layers)
            color_layers = blend * fg.unsqueeze(-4) + (1 - blend) * bg.unsqueeze(-4)
        else:
            color_layers = prev_images[:, -1].unsqueeze(-4).expand(bs, self.n_plane, self.n_channels, *self.img_size)

        rgba_layers = torch.cat([color_layers, alpha_layers], dim=-3)  # (b * t, d, c + 1, h, w)

        mpi_features.update(
            alpha_layers=alpha_layers,
            color_layers=color_layers,
            rgba_layers=rgba_layers,
        )

        """ flow prediction """
        # pool with transmittance
        transmittance = mpi.layer_weights(alpha_layers)
        if self.normalize_theta_weight:
            norm = transmittance.flatten(-2, -1).sum(-1)
            normalized_transmittance = transmittance / norm[..., None, None]
            theta = theta * normalized_transmittance
            theta = theta.sum((-2, -1))
        else:
            theta = (theta * transmittance).mean((-2, -1))
        theta = theta * self.theta_output_multiplier
        theta = theta.unflatten(-1, (2, 3))
        theta_layers = theta + self.theta_bias  # (b, d, 2, 3)

        mpi_features.update(
            theta_layers=theta_layers,
        )

        if self.affine_grid_on_cpu:
            coordinates = F.affine_grid(
                batch_d(theta_layers).cpu(),
                [bs * self.n_plane, self.n_channels + 1, *self.img_size],
                align_corners=True,
            ).cuda()
        else:
            coordinates = F.affine_grid(
                batch_d(theta_layers),
                [bs * self.n_plane, self.n_channels + 1, *self.img_size],
                align_corners=True,
            )
        flow_layers = unbatch_d(grid_to_flow(coordinates))

        pred_flow = mpi.compose_back_to_front(torch.cat([flow_layers, alpha_layers], dim=-3))
        coordinates = unbatch_d(coordinates)
        mpi_features.update(coordinates=coordinates,
                            flow_layers=flow_layers,
                            pred_flow=pred_flow)

        """ apply warping """

        if self.grid_sample_on_cpu:
            warped_rgba_layers = F.grid_sample(
                input=batch_d(rgba_layers).cpu(),
                grid=batch_d(coordinates).cpu(),
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            ).cuda()
        else:
            warped_rgba_layers = F.grid_sample(
                input=batch_d(rgba_layers),
                grid=batch_d(coordinates),
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            )
        warped_rgba_layers = unbatch_d(warped_rgba_layers)

        if self.back_depth == -1 and self.front_depth == -1:
            disparity_linspace = torch.stack([torch.linspace(1 / gt_features['bds'][b_ind].item(), 1 / gt_features['fds'][b_ind].item(), self.n_plane) for b_ind in range(bs)], dim=0).cuda()
            assert disparity_linspace.shape == (bs, self.n_plane)
            mpi_features['disparity_linspace'] = disparity_linspace
            disparity_layers = disparity_linspace.view(bs, self.n_plane, 1, 1, 1).expand(bs, self.n_plane, 1, *self.img_size)
        else:
            disparity_linspace = torch.linspace(1 / self.back_depth, 1 / self.front_depth, self.n_plane).to(prev_images)
            mpi_features['disparity_linspace'] = disparity_linspace.view(1, self.n_plane).expand(bs, self.n_plane)
            disparity_layers = disparity_linspace.view(1, self.n_plane, 1, 1, 1).expand(bs, self.n_plane, 1, *self.img_size)

        # it does not make sense to use depth layers
        # because inverse disparity layers then compose is different from compose disparity layers then inverse
        # depth_layers = 1 / disparity_layers

        pred_disparity = mpi.compose_back_to_front(torch.cat([disparity_layers, alpha_layers], dim=-3))
        warped_alpha_layers = warped_rgba_layers[:, :, -1:, :, :]
        pred_disparity_next = mpi.compose_back_to_front(torch.cat([disparity_layers, warped_alpha_layers], dim=-3))

        mpi_features.update(warped_rgba_layers=warped_rgba_layers,  # I_{t+1} rgba layers
                            fake_images=mpi.compose_back_to_front(warped_rgba_layers),  # I_{t+1}
                            recon_image=mpi.compose_back_to_front(rgba_layers),  # I_t
                            # depth_layers=depth_layers,
                            disparity_layers=disparity_layers,
                            pred_disparity=pred_disparity,  # I_t depth map
                            # pred_depth=1 / pred_disparity,
                            pred_disparity_next=pred_disparity_next,  # I_{t+1} depth map
                            # pred_depth_next=1 / pred_disparity_next,
                            )

        return mpi_features
