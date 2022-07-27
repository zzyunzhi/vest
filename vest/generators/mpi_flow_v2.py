import torch.nn as nn
from vest.losses.stereo import StereoLoss, BaseLoss
from tu.configs import AttrDict
from vest.flows.mpi_single_view_net_flow_unet_color_blend_v2 import MPIEmbedder
import torch.nn.functional as F


class MPILoss(BaseLoss):
    @property
    def criteria(self):
        return dict(
            recon_prev_image=self.compute_recon_prev_image_loss,
        )

    @staticmethod
    def compute_recon_prev_image_loss(data_t, gen_out_t):
        loss = dict(
            l1=F.l1_loss(gen_out_t['recon_image'], data_t['prev_images'][:, -1]),
            l2=F.mse_loss(gen_out_t['recon_image'], data_t['prev_images'][:, -1]),
        )

        return loss, dict()


class Generator(nn.Module):
    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.n_plane = gen_cfg.embed.n_plane

        # define mpi network
        assert gen_cfg.embed.type == 'vest.flows.mpi_single_view_net_flow_unet_color_blend_v2', gen_cfg.embed.type
        self.mpi_embedding = MPIEmbedder(gen_cfg.embed, data_cfg)
        self.mpi_back_depth = self.mpi_embedding.back_depth
        self.mpi_front_depth = self.mpi_embedding.front_depth

        assert not gen_cfg.use_pts_transformer
        self.use_flow_loss = gen_cfg.use_flow_net or gen_cfg.use_pwc_net or gen_cfg.use_eulerian_motion
        assert not self.use_flow_loss
        self.use_depth_net = gen_cfg.use_mono or gen_cfg.use_midas or gen_cfg.use_monodepth2
        assert not self.use_depth_net
        self.use_stereo = gen_cfg.use_stereo
        self.use_stereo_detached = gen_cfg.use_stereo_detached
        self.use_stereo_forward = gen_cfg.use_stereo_forward
        if self.use_stereo or self.use_stereo_detached or self.use_stereo_forward:
            self.stereo_loss = StereoLoss(cfg=AttrDict(gen=gen_cfg, data=data_cfg))
        self.use_stereo_ssi = gen_cfg.use_stereo_ssi
        assert not self.use_stereo_ssi
        self.mpi_loss = MPILoss()

    def forward(self, data, compute_loss=False):
        # output = self.core(data)
        gt_features = dict()
        if 'fds' in data:
            gt_features['fds'] = data['fds']
            gt_features['bds'] = data['bds']
        output = self.mpi_embedding(data['prev_images'], gt_features)
        output.update(loss=dict(), auxiliary=dict())
        if compute_loss:
            loss, auxiliary = self.mpi_loss.compute_loss(gen_out_t=output, data_t=data)
            output['loss'].update(loss)
            output['auxiliary'].update(auxiliary)

            if self.use_stereo_forward and data['tp1'] == 2:#== data['prev_images'].shape[1]:
                # no need to run stereo_output
                loss, aux = self.stereo_loss.compute_loss(
                    data_t=data,
                    gen_out_t=output,
                    # stereo_gen_out_t=None,
                    # auxiliary=output['auxiliary'],
                )
                output['loss'].update(loss)
                output['auxiliary'].update(aux)
        return output
