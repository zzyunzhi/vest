from vest.layers.monodepth2 import SSIM
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vest.utils.mpi as mpi
from vest.third_party.mine.operations import rendering_utils
from vest.third_party.deep3d.train_mpi.projector import projective_forward_warp


class BaseLoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    @property
    def criteria(self):
        raise NotImplementedError

    def compute_loss(self, prefix='', **kwargs):
        loss = dict()
        auxiliary = dict()
        if prefix:
            prefix += '_'

        for name, criterion in self.criteria.items():
            loss_this, aux = criterion(**kwargs)
            if isinstance(loss_this, dict):
                loss.update({f"{prefix}{name}_{k}": v for k, v in loss_this.items()})
            else:
                loss[f"{prefix}{name}"] = loss_this

            for k in aux.keys():
                if f"{prefix}{k}" in auxiliary:
                    print(f"{prefix}{k} from criterion {name} overwrites previous")
            auxiliary.update({f"{prefix}{k}": v for k, v in aux.items()})

        return loss, auxiliary


def compute_scale_factor(disparity_syn_pt3dsrc, pt3d_disp_src):
    assert disparity_syn_pt3dsrc.shape == pt3d_disp_src.shape
    # output shape (bs,)
    # B = pt3d_disp_src.size()[0]

    # 1. calibrate the scale between the src image/depth and our synthesized image/depth
    scale_factor = torch.exp(torch.mean(
        torch.log(disparity_syn_pt3dsrc) - torch.log(pt3d_disp_src),
        dim=2, keepdim=False)).squeeze(1)  # B
    return scale_factor


class StereoLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_disp_scale = cfg.gen.use_disp_scale
        self.use_disp_scale_detached = cfg.gen.use_disp_scale_detached
        assert not (self.use_disp_scale and self.use_disp_scale_detached)
        self.ssim = SSIM().cuda()

    @property
    def criteria(self):
        return dict(
            stereo_forward=self.compute_forward_consistency,
        )
    def compute_forward_consistency_kitti(self, data_t, gen_out_t):
        loss, aux = dict(), dict()

        # camera calibration is consistent for frames from the same clip
        intrinsics = data_t[('K', 0)]
        assert intrinsics.shape[1:] == (4,), intrinsics.shape[1:]
        target_intrinsics = intrinsics
        pose = torch.Tensor(np.eye(3, 4)).cuda()
        target_pose = data_t['stereo_T']
        assert pose.shape == target_pose.shape[1:]

        depth_layers = torch.reciprocal(gen_out_t['disparity_linspace'])

        def update_for(src_layers, trg_rgb, suffix=''):
            trg_layers = mpi.render_layers(
                layers=src_layers.movedim(-3, -1), depths=depth_layers,
                pose=pose,
                intrinsics=intrinsics,
                target_pose=target_pose,
                target_intrinsics=target_intrinsics,
                clamp=True,
            ).movedim(-1, -3)
            trg_rgb_syn = mpi.compose_back_to_front(trg_layers)
            this_loss = self.compute_loss_items(
                pred=trg_rgb_syn,
                target=trg_rgb,
            )
            loss.update({f"{k}{suffix}": v.mean() for k, v in this_loss.items()})

            aux.update({
                f"trg_rgba{suffix}": trg_layers,
                f"trg_rgb_syn{suffix}": trg_rgb_syn,
            })

        update_for(
            src_layers=gen_out_t['rgba_layers'],
            trg_rgb=data_t['stereo_prev_images'][:, -1],
            suffix='_t'
        )
        update_for(
            src_layers=gen_out_t['warped_rgba_layers'],
            trg_rgb=data_t['stereo_image'],
            suffix='_tp1'
        )

        return loss, aux

    def compute_forward_consistency_mvcam(self, data_t, gen_out_t):
        loss, aux = dict(), dict()

        # camera calibration is consistent for frames from the same clip
        src_int = data_t['src_K']
        tgt_int = data_t['tgt_K']

        src_ext = data_t['src_w2c']
        tgt_ext = data_t['tgt_w2c']
        transforms = torch.matmul(tgt_ext, torch.inverse(src_ext))

        def update_for(src_layers, trg_rgb, depths, suffix=''):
            # trg_layers = projective_forward_warp(src_layers.movedim(-3, -1), src_int, tgt_int,
            #                                      transforms, depths).movedim(-1, -3)
            # (b, d, c, h, w) -> (d, b, h, w, c)
            trg_layers = projective_forward_warp(src_layers.permute(1, 0, 3, 4, 2), src_int, tgt_int,
                                                 transforms, depths).permute(1, 0, 4, 2, 3)
            trg_rgb_syn = mpi.compose_back_to_front(trg_layers)
            this_loss = self.compute_loss_items(
                pred=trg_rgb_syn,
                target=trg_rgb,
            )
            loss.update({f"{k}{suffix}": v.mean() for k, v in this_loss.items()})

            aux.update({
                f"trg_rgba{suffix}": trg_layers,
                f"trg_rgb_syn{suffix}": trg_rgb_syn,
            })

        if self.use_disp_scale:
            loss_disp, aux_disp = self.compute_scale_factor(data_t, gen_out_t)
            loss.update(loss_disp)
            update_for(
                src_layers=gen_out_t['rgba_layers'],
                trg_rgb=data_t['stereo_prev_images'][:, -1],
                depths=torch.reciprocal(gen_out_t['disparity_linspace']) * aux_disp['scale_factor'].view(-1, 1),
                suffix='_t'
            )
            aux.update(aux_disp)
            return loss, aux
        if self.use_disp_scale_detached:
            loss_disp, aux_disp = self.compute_scale_factor(data_t, gen_out_t)
            loss.update(loss_disp)
            update_for(
                src_layers=gen_out_t['rgba_layers'],
                trg_rgb=data_t['stereo_prev_images'][:, -1],
                depths=torch.reciprocal(gen_out_t['disparity_linspace']) * aux_disp['scale_factor'].clone().detach().view(-1, 1),
                suffix='_t'
            )
            aux.update(aux_disp)
            return loss, aux

        update_for(
            src_layers=gen_out_t['rgba_layers'],
            trg_rgb=data_t['stereo_prev_images'][:, -1],
            depths=torch.reciprocal(gen_out_t['disparity_linspace']),
            suffix='_t',
        )

        return loss, aux

    def compute_scale_factor(self, data_t, gen_out_t):
        loss = dict()
        pt3d_src_disp_min = []
        pt3d_src_disp_max = []
        scale_factor = []
        loss_disp = []
        sparse_depth_map = []
        bs = len(data_t['depths'])
        for b_ind in range(bs):
            pt3d_src_this = data_t['depths'][b_ind]  # (n,)
            pt3d_mask = pt3d_src_this >= 1e-3  # 1  # (n,)
            n_pts = pt3d_mask.sum().item()
            if not pt3d_mask.all():
                print('[ERROR] encountered depth < 1e-4', n_pts, pt3d_src_this.shape[-1])

            pt3d_src_this = pt3d_src_this[pt3d_mask]  # (n')
            pt3d_src_this = pt3d_src_this.view(1, 1, n_pts)
            src_pt3d_disp = torch.reciprocal(pt3d_src_this)

            src_pt3d_pxpy = data_t['xys_cam'][b_ind]  # (2, n)
            src_pt3d_pxpy = src_pt3d_pxpy[:, pt3d_mask]  # (2, n')
            src_pt3d_pxpy = src_pt3d_pxpy.view(1, 2, n_pts)

            src_disparity_syn_this = gen_out_t['pred_disparity'][b_ind:b_ind+1]
            src_pt3d_disp_syn = rendering_utils.gather_pixel_by_pxpy(src_disparity_syn_this, src_pt3d_pxpy)  # Bx1xN_pt
            assert src_pt3d_disp_syn.shape == (1, 1, n_pts)

            scale_factor_this = compute_scale_factor(src_pt3d_disp_syn, src_pt3d_disp)  # (1,)
            if torch.isnan(scale_factor_this):
                print('[ERROR] THIS SHOULD NOT HAPPEN')
                scale_factor_this[torch.isnan(scale_factor_this)] = 1
            scale_factor.append(scale_factor_this)

            # disparity at src frame
            src_pt3d_disp_syn_scaled = src_pt3d_disp_syn / scale_factor_this.view(1, 1, 1)
            loss_disp_pt3dsrc_this = torch.abs(
                torch.log(src_pt3d_disp_syn_scaled) - torch.log(src_pt3d_disp)).squeeze(-2).mean(-1)
            loss_disp.append(loss_disp_pt3dsrc_this)
            pt3d_src_disp_min.append(src_pt3d_disp.squeeze(-2).min(-1).values)
            pt3d_src_disp_max.append(src_pt3d_disp.squeeze(-2).max(-1).values)

            # get sparse depth map
            h, w = data_t['image'].shape[-2:]
            pxpy = src_pt3d_pxpy  # (1, 2, n)
            pxpy_int = torch.round(pxpy).to(torch.int64)
            pxpy_int[:, 0, :] = torch.clamp(pxpy_int[:, 0, :], min=0, max=w - 1)
            pxpy_int[:, 1, :] = torch.clamp(pxpy_int[:, 1, :], min=0, max=h - 1)
            pxpy_idx = pxpy_int[:, 0, :] + w * pxpy_int[:, 1, :]  # (b, n)
            flatdm = torch.zeros((1, h * w)).cuda()  # (
            flatz = pt3d_src_this.view(1, n_pts)  # b,N
            flatdm.scatter_(dim=1, index=pxpy_idx, src=flatz)
            sparse_dm_this = flatdm.view(1, 1, h, w)
            sparse_depth_map.append(sparse_dm_this)

        scale_factor = torch.cat(scale_factor, dim=0)  # (B,)
        sparse_depth_map = torch.cat(sparse_depth_map, dim=0)  # (b, 1, h, w)
        loss_disp = torch.cat(loss_disp, dim=0)  # (B,)
        pt3d_src_disp_min = torch.cat(pt3d_src_disp_min, dim=0)  # (B,)
        pt3d_src_disp_max = torch.cat(pt3d_src_disp_max, dim=0)  # (B,)
        assert scale_factor.shape == (bs,), scale_factor.shape
        assert loss_disp.shape == (bs,), loss_disp.shape
        assert pt3d_src_disp_max.shape == (bs,), pt3d_src_disp_max.shape
        loss['l1_sparse_disparity'] = loss_disp.mean()  # FIXME: detach scale_factor?
        loss['pt3d_src_disp_min'] = pt3d_src_disp_min.mean()
        loss['pt3d_src_disp_max'] = pt3d_src_disp_max.mean()
        loss['scale_mean'] = scale_factor.mean()
        loss['scale_min'] = scale_factor.min()
        loss['scale_max'] = scale_factor.max()

        return loss, dict(scale_factor=scale_factor, colmap_sparse_disp_map=torch.reciprocal(sparse_depth_map))

    def compute_forward_consistency(self, data_t, gen_out_t):
        if 'stereo_T' in data_t:
            return self.compute_forward_consistency_kitti(data_t, gen_out_t)
        if 'src_w2c' in data_t:
            return self.compute_forward_consistency_mvcam(data_t, gen_out_t)
        # raise NotImplementedError(data_t.keys())
        # source image comes from main view
        # depth map comes from main view
        # project to stereo view

        layers = gen_out_t['rgba_layers']
        bs, d, _, *img_size = layers.shape

        loss, aux = dict(), dict()
        if 'src_pose' in data_t and 'src_xyzs' not in data_t:
            # mvcam colmap format
            target_stereo_view = data_t['query_image']
            intrinsics = data_t['src_camera']  # (b, 4)
            pose = data_t['src_pose']  # (b, 3, 4)
            target_intrinsics = data_t['query_camera']  # (b, 4)
            target_pose = data_t['query_pose']  # (b, 3, 4)
            # adjust with depth bounds
            # src_disparity_syn = gen_out_t['pred_disparity']

            # depth used = torch.reciprocal(gen_out_t['disparity_layers']) * scale_factor.view(bs, 1, 1, 1, 1))

            scale_factor = torch.ones((bs,)).cuda()
            # # or
            # src_disparity_gt_min = 1 / data_t['bds']
            # scale_factor = src_disparity_gt_min / torch.quantile(src_disparity_syn.flatten(start_dim=1), q=0.99, dim=1)
            # # or
            # src_disparity_gt_max = 1 / data_t['fds']
            # scale_factor = src_disparity_gt_max / torch.quantile(src_disparity_syn.flatten(start_dim=1), q=0.01, dim=1)

            sparse_depth_map = torch.ones((bs, 1, *img_size)).cuda()

        elif 'src_pose' in data_t:  # estate
            target_stereo_view = data_t['query_image']

            assert 'stereo_T' not in data_t
            intrinsics = data_t['src_camera']  # (b, 4)
            pose = data_t['src_pose']  # (b, 3, 4)
            target_intrinsics = data_t['query_camera']  # (b, 4)
            target_pose = data_t['query_pose']  # (b, 3, 4)

            src_disparity_syn = gen_out_t['pred_disparity']
            # (b, 1, h, w)

            assert isinstance(data_t['src_xyzs'], list)
            pt3d_src_disp_min = []
            pt3d_src_disp_max = []
            scale_factor = []
            loss_disp = []
            sparse_depth_map = []
            # compute scale factor
            for b_ind in range(bs):
                # pt3d_src_this = data_t['src_xyzs'][b_ind].unsqueeze(0)
                # pt3d_src_this = pt3d_src_this.clamp(min=1)
                # pt3d_src_this = pt3d_src_this[pt3d_src_this[:, 2:, :] >= 1]  # FIXME: ?
                pt3d_src_this = data_t['src_xyzs'][b_ind]  # (3, n)
                pt3d_mask = pt3d_src_this[2, :] >= 1e-4  # 1  # (n,)
                if (~pt3d_mask).any():
                    print('[ERROR] encountered depth < 1e-4', (~pt3d_mask).sum().item(), pt3d_src_this.shape[-1])
                pt3d_src_this = pt3d_src_this[:, pt3d_mask]  # (3, n')
                pt3d_src_this = pt3d_src_this.unsqueeze(0)

                src_pt3d_pxpy = data_t['src_xys_cam'][b_ind]  # (2, n)
                src_pt3d_pxpy = src_pt3d_pxpy[:, pt3d_mask]  # (2, n')
                src_pt3d_pxpy = src_pt3d_pxpy.unsqueeze(0)

                src_disparity_syn_this = src_disparity_syn[b_ind:b_ind+1]

                src_pt3d_disp = torch.reciprocal(pt3d_src_this[:, 2:, :])  # Bx1xN_pt
                # if False:
                #     K_src_scaled_this = K_src_scaled[b_ind:b_ind + 1]
                #     src_pt3d_pxpy = torch.matmul(K_src_scaled_this, pt3d_src_this)  # Bx3x3 * Bx3xN_pt -> Bx3xN_pt
                #     src_pt3d_pxpy = src_pt3d_pxpy[:, 0:2, :] / src_pt3d_pxpy[:, 2:, :]  # Bx2xN_pt
                src_pt3d_disp_syn = rendering_utils.gather_pixel_by_pxpy(src_disparity_syn_this, src_pt3d_pxpy)  # Bx1xN_pt
                assert src_pt3d_disp_syn.shape[0] == 1 and src_pt3d_disp.shape[0] == 1
                scale_factor_this = compute_scale_factor(src_pt3d_disp_syn, src_pt3d_disp)  # (B=1,)
                assert not torch.isnan(scale_factor_this)
                scale_factor.append(scale_factor_this)

                # disparity at src frame
                src_pt3d_disp_syn_scaled = src_pt3d_disp_syn / scale_factor_this.view(1, 1, 1)
                loss_disp_pt3dsrc_this = torch.abs(
                    torch.log(src_pt3d_disp_syn_scaled) - torch.log(src_pt3d_disp)).squeeze(-2).mean(-1)
                loss_disp.append(loss_disp_pt3dsrc_this)
                pt3d_src_disp_min.append(src_pt3d_disp.squeeze(-2).min(-1).values)
                pt3d_src_disp_max.append(src_pt3d_disp.squeeze(-2).max(-1).values)

                # get sparse depth map
                h, w = data_t['image'].shape[-2:]
                pxpy = src_pt3d_pxpy  # (1, 2, n)
                pxpy_int = torch.round(pxpy).to(torch.int64)
                pxpy_int[:, 0, :] = torch.clamp(pxpy_int[:, 0, :], min=0, max=w - 1)
                pxpy_int[:, 1, :] = torch.clamp(pxpy_int[:, 1, :], min=0, max=h - 1)
                pxpy_idx = pxpy_int[:, 0, :] + w * pxpy_int[:, 1, :]  # (b, n)
                flatdm = torch.zeros((1, h * w)).cuda()  # (
                flatz = pt3d_src_this[:, 2, :]  # b,N
                flatdm.scatter_(dim=1, index=pxpy_idx, src=flatz)
                sparse_dm_this = flatdm.view(1, 1, h, w)
                sparse_depth_map.append(sparse_dm_this)

            # print("[DEBUG] scale factor", [t.item() for t in scale_factor])
            scale_factor = torch.cat(scale_factor, dim=0)  # (B,)
            loss_disp = torch.cat(loss_disp, dim=0)  # (B,)
            pt3d_src_disp_min = torch.cat(pt3d_src_disp_min, dim=0)  # (B,)
            pt3d_src_disp_max = torch.cat(pt3d_src_disp_max, dim=0)  # (B,)
            sparse_depth_map = torch.cat(sparse_depth_map, dim=0)  # (B,)
            if os.getenv('CHECK') == '1':
                assert scale_factor.shape == (bs,), scale_factor.shape
                assert loss_disp.shape == (bs,), loss_disp.shape
                assert pt3d_src_disp_max.shape == (bs,), pt3d_src_disp_max.shape
            loss['l1_sparse_disparity'] = loss_disp.mean()  # FIXME: detach scale_factor?
            loss['pt3d_src_disp_min'] = pt3d_src_disp_min.mean()
            loss['pt3d_src_disp_max'] = pt3d_src_disp_max.mean()
            loss['scale_mean'] = scale_factor.mean()
            loss['scale_min'] = scale_factor.min()
            loss['scale_max'] = scale_factor.max()

        elif 'stereo_T' in data_t:  # kitti
            raise NotImplementedError()
            target_stereo_view = data_t['stereo_prev_images'][:, -1]

            intrinsics = data_t[('K', 0)]
            # normalize intrinsics
            if len(intrinsics.shape) == 3:
                img_size = layers.shape[-2:]
                intrinsics = torch.stack([intrinsics[:, 0, 0] / img_size[1], intrinsics[:, 1, 1] / img_size[0],
                                          intrinsics[:, 0, 2] / img_size[1], intrinsics[:, 1, 2] / img_size[0]],
                                         dim=-1).cuda()
                target_intrinsics = intrinsics
                pose = torch.from_numpy(np.eye(3, 4)).float().cuda()
                target_pose = data_t['stereo_T'][:, :3, :]
            else:
                assert intrinsics.shape[1:] == (4,), intrinsics.shape[1:]
                target_intrinsics = intrinsics
                pose = torch.from_numpy(np.eye(3, 4)).float().cuda()
                target_pose = data_t['stereo_T']
                assert pose.shape == target_pose.shape[1:]

        def update_this(name, depth_layers):
            depth_layers = depth_layers.flatten(start_dim=-2).mean(-1).squeeze(-1)  # remove spatial dim, assume spatially uniform values
            new_layers = mpi.render_layers(layers=layers.movedim(-3, -1), depths=depth_layers,
                                           pose=pose,
                                           intrinsics=intrinsics,
                                           target_pose=target_pose,
                                           target_intrinsics=target_intrinsics,
                                           clamp=True,   # !!!!!! FIXME: should it be true or false? or return valid mask
                                           ).movedim(-1, -3)
            syn_stereo_view = mpi.compose_back_to_front(new_layers)

            this_loss = self.compute_loss_items(
                pred=syn_stereo_view,
                target=target_stereo_view,
            )
            loss.update({f"{k}_{name}": v.mean() for k, v in this_loss.items()})
            aux.update(
                {
                    # for I_t
                    f"syn_forward_stereo_view_{name}": syn_stereo_view,  # reprojected main view layers, then composed
                    f"syn_forward_stereo_view_before_compose_{name}": new_layers,
                    "trg_rgba_t": new_layers,
                    "trg_rgb_syn_t": syn_stereo_view,
                }
            )

        if self.use_disp_scale:
            update_this('mpi_inv_disp', torch.reciprocal(gen_out_t['disparity_layers']) * scale_factor.view(bs, 1, 1, 1, 1))
            aux.update(dict(scale_factor=scale_factor, colmap_sparse_disp_map=torch.reciprocal(sparse_depth_map)))
        elif self.use_disp_scale_detached:
            update_this('mpi_inv_disp', torch.reciprocal(gen_out_t['disparity_layers']) * scale_factor.clone().detach().view(bs, 1, 1, 1, 1))
            aux.update(dict(scale_factor=scale_factor, colmap_sparse_disp_map=torch.reciprocal(sparse_depth_map)))
        else:
            update_this('mpi_inv_disp', torch.reciprocal(gen_out_t['disparity_layers']))
        # update_this('mpi_inv_disp', torch.reciprocal(gen_out_t['disparity_layers']))
        # if self.no_grad_for_inv_disp_scaled:
        #     with torch.no_grad():
        #         update_this('mpi_inv_disp_scaled', 1 / gen_out_t['disparity_layers_scaled'])
        # else:
        if False:
            update_this('mpi_inv_disp_scaled', 1 / gen_out_t['disparity_layers_scaled'])

        return loss, aux

    def compute_loss_items(self, pred, target):
        assert pred.shape == target.shape, (pred.shape, target.shape)
        assert len(pred.shape) == 4, pred.shape

        # # cannot use F.l1_loss because don't want stop gradient for target
        # abs_diff = torch.abs(target - pred)
        # l1_loss = abs_diff.mean(1, True)
        l1_loss = F.l1_loss(input=pred, target=target)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        # reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        loss = dict(l1=l1_loss, ssim=ssim_loss, #reprojection=reprojection_loss,
                    # perceptual=self.perceptual(pred_img=pred, gt_img=target),
                    )

        return loss

