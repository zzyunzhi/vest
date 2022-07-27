import warnings

import torch
from tqdm import tqdm
import copy
import lpips
from PIL import Image
import vest.utils.mpi as mpi
import os
from vest.utils.visualization.html_table import HTMLTableColumnDesc, HTMLTableVisualizer
from vest.utils.misc import to_cuda
from vest.model_utils.fs_vid2vid import (concat_frames)
from vest.utils.distributed import is_master
from vest.utils.meters import Meter
from vest.trainers.utils import visualize_rgba, stack_layouts, get_denormalize_rgb
from tu.loggers.utils import collect_tensor
from tu.flow import tensor2flow
from vest.utils.visualization.html_helper import BaseHTMLHelper
import torch.nn.functional as F
from vest.trainers.vid2vid import Trainer as BaseTrainer
from vest.flows.grid_utils import grid_to_flow, compute_flow_conf, grid_to_norm_flow, flow_to_norm_flow
from vest.losses.similarity import PNSR, SSIMMetric, LPIPSVGGVideoAutoencoder


class Trainer(BaseTrainer):
    def _define_custom_losses(self):
        r"""All other custom losses are defined here."""
        self.use_flow = False
        self.weights.update(self.cfg.trainer.custom_loss_weight)

        self.use_pts_transformer = self.cfg.gen.use_pts_transformer
        assert not self.use_pts_transformer
        self.use_stereo = self.cfg.gen.use_stereo
        self.use_stereo_detached = self.cfg.gen.use_stereo_detached
        self.use_stereo_forward = self.cfg.gen.use_stereo_forward
        self.use_stereo_ssi = self.cfg.gen.use_stereo_ssi
        assert not self.cfg.trainer.model_average
        assert not self.use_stereo
        assert not self.use_stereo_detached
        assert not self.use_stereo_ssi
        assert self.use_stereo_forward

        self.similarity_metrics = dict(ssim=SSIMMetric(), pnsr=PNSR())
        if 'estate' in self.cfg.data.name:
            # for runs after Nov 9 11:30 pm
            self.similarity_metrics.update(lpips_vgg=LPIPSVGGVideoAutoencoder())
        else:
            self.similarity_metrics.update(lpips_vgg=lpips.LPIPS(net='vgg'), lpips_alex=lpips.LPIPS(net='alex'))
        for v in self.similarity_metrics.values():
            v.cuda()

        self._assign_criteria('L1', torch.nn.L1Loss(), self.cfg.trainer.loss_weight.L1)

        # validation
        self.val_gen_losses = dict()
        # self.losses.update(val=self.val_gen_losses)

    def gen_update(self, data):
        data_t = self.get_data_t(data, None, None, self.sequence_length-1)
        net_G_output = self.net_G(data_t, compute_loss=True)
        self.get_gen_losses(data_t, net_G_output, {})

    def gen_frames(self, data):
        r"""Generate a sequence of frames given a sequence of data.

        Args:
            data (dict): Training data at the current iteration.
            use_model_average (bool): Whether to use model average
                for update or not.
        """
        net_G_output = None  # Previous generator output.
        data_prev = None  # Previous data.
        net_G = self.net_G

        # Iterate through the length of sequence.
        all_info = {'inputs': [], 'outputs': []}
        for t in range(self.cfg.data.num_frames_G-1, self.sequence_length):
            # Get the data at the current time frame.
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Generator forward.
            with torch.no_grad():
                net_G_output = net_G(data_t, compute_loss=True)

            if t == self.cfg.data.num_frames_G-1:
                # Get the output at beginning of sequence for visualization.
                first_net_G_output = net_G_output

            all_info['inputs'].append(data_t)
            all_info['outputs'].append(net_G_output)

        return first_net_G_output, net_G_output, all_info

    def get_gen_losses(self, data_t, net_G_output, net_D_output):
        r"""Compute generator losses.

        Args:
            data_t (dict): Training data at the current time t.
            net_G_output (dict): Output of the generator.
            net_D_output (dict): Output of the discriminator.
        """
        self.opt_G.zero_grad()

        self._get_gen_losses(data_t, net_G_output, net_D_output)
        for k, v in self.gen_losses.items():
            if torch.isnan(v):
                print(self.gen_losses)
                import ipdb; ipdb.set_trace()
                raise RuntimeError

        total_loss = self.gen_losses['total']
        total_loss.backward()

        self.opt_G.step()

    def _get_gen_losses(self, data_t, net_G_output, net_D_output, compute_total=True):

        if net_D_output:
            raise NotImplementedError

        # Perceptual loss.
        if net_G_output['fake_images'].shape[-1] < 224:
            # weird upsampling
            fake_image = F.interpolate(
                net_G_output['fake_images'], size=(224, 224),
                mode='bilinear', align_corners=False,
            )
            image = F.interpolate(
                data_t['image'], size=(224, 224),
                mode='bilinear', align_corners=False,
            )
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](fake_image, image)

        else:
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                net_G_output['fake_images'], data_t['image'])

        if self.use_stereo or self.use_stereo_detached or self.use_stereo_forward or self.use_stereo_ssi:
            if 'stereo_prev_images' in data_t:
                trg_syn_image = data_t['stereo_prev_images'][:, -1]
            else:
                trg_syn_image = data_t['query_image']
            if self.use_stereo_ssi:
                syn_image = net_G_output['auxiliary']['trg_rgb_syn']
            elif 'syn_forward_stereo_view_mpi_inv_disp' in net_G_output['auxiliary']:
                syn_image = net_G_output['auxiliary']['syn_forward_stereo_view_mpi_inv_disp']
            else:
                syn_image = net_G_output['auxiliary']['trg_rgb_syn_t']

            # Perceptual loss for view synthesis
            self.gen_losses['stereo_forward_perc_mpi_inv_disp'] = self.criteria['Perceptual'](
                syn_image, trg_syn_image,
            )

        # L1 loss.
        self.gen_losses['L1'] = self.criteria['L1'](
            net_G_output['fake_images'], data_t['image'])

        assert 'raw' not in net_D_output
        assert self.add_dis_cfg is None
        assert not self.use_flow
        assert self.cfg.trainer.loss_weight.temporal_gan == 0

        # Other custom losses.
        self._get_custom_gen_losses(data_t, net_G_output, net_D_output)

        if not compute_total:
            self.gen_losses['total'] = torch.zeros(()).cuda()

            # for key in self.gen_losses:
            #     if key != 'total' and key not in self.weights:
            #         print('\n   WARNINGS: missing', key, 'from self.weights, set to 0')
            #         self.weights[key] = 0
            return

        # Sum all losses together.
        total_loss = self.Tensor(1).fill_(0)
        for key in self.gen_losses:
            if key != 'total' and self.weights[key] > 0:  # to exclude inf
                total_loss += self.gen_losses[key] * self.weights[key]

        for key in self.weights:
            if self.weights[key] > 0 and key not in self.gen_losses:
               raise RuntimeError('missing key with positive loss weight', key, self.weights[key], self.gen_losses.keys())

        self.gen_losses['total'] = total_loss

    def _get_custom_gen_losses(self, data_t, net_G_output, net_D_output):
        r"""All other custom generator losses go here.

        Args:
            data_t (dict): Training data at the current time t.
            net_G_output (dict): Output of the generator.
            net_D_output (dict): Output of the discriminator.
        """
        for name, loss_item in net_G_output['loss'].items():
            self.gen_losses[name] = loss_item

        if 'fake_image_initial' in net_G_output:
            self.gen_losses['perceptual_initial'] = self.criteria['Perceptual'](
                net_G_output['fake_image_initial'], data_t['image']
            )
            self.gen_losses['l1_initial'] = self.criteria['L1'](
                net_G_output['fake_image_initial'], data_t['image']
            )
        if 'fake_image_refined' in net_G_output:
            self.gen_losses['Perceptual_refined'] = self.criteria['Perceptual'](
                net_G_output['fake_image_refined'], data_t['image']
            )
            self.gen_losses['L1_refined'] = self.criteria['L1'](
                net_G_output['fake_image_refined'], data_t['image']
            )

        # evaluation metrics

        with torch.no_grad():

            denormalize = get_denormalize_rgb(self.cfg)

            # compare next_frame prediction quality of gt flow and mpi

            # compute ssim between gt_warp_out and real I_t
            # this is to compare which of gt_warp_out and fake I_t is more similar is real I_t
            # note that gt_warp_out receives information from real I_t, it's a performance upper bound
            for name, criterion in self.similarity_metrics.items():
                if self.net_G.module.use_flow_loss:
                    self.gen_losses[f'eval_{name}_pred_from_flownet'] = criterion(net_G_output['auxiliary']['gt_warp_out'], data_t['image']).mean()
                self.gen_losses[f'eval_{name}_pred_from_mpi'] = criterion(
                    denormalize(net_G_output['fake_images']), denormalize(data_t['image'])).mean()

            if self.use_stereo or self.use_stereo_detached or self.use_stereo_forward or self.use_stereo_ssi:
                if 'stereo_prev_images' in data_t:
                    trg_syn_image = data_t['stereo_prev_images'][:, -1]
                else:
                    trg_syn_image = data_t['query_image']
                if self.use_stereo_ssi:
                    syn_image = net_G_output['auxiliary']['trg_rgb_syn']
                elif 'syn_forward_stereo_view_mpi_inv_disp' in net_G_output['auxiliary']:
                    syn_image = net_G_output['auxiliary']['syn_forward_stereo_view_mpi_inv_disp']
                else:
                    syn_image = net_G_output['auxiliary']['trg_rgb_syn_t']

                # denormalize to [0, 1]!
                trg_syn_image = denormalize(trg_syn_image)
                syn_image = denormalize(syn_image)

                for name, criterion in self.similarity_metrics.items():

                    """ project main view to stereo view """

                    self.gen_losses[f'eval_{name}_syn_forward_stereo_view_mpi_inv_disp'] = \
                        criterion(syn_image, trg_syn_image).mean()

    def get_data_t(self, data, net_G_output, data_prev, t):
        r"""Get data at current time frame given the sequence of data.

        Args:
            data (dict): Training data for current iteration.
            net_G_output (dict): Output of the generator (for previous frame).
            data_prev (dict): Data for previous frame.
            t (int): Current time.
        """
        image = data['images'][:, t]

        num_frames_G = self.cfg.data.num_frames_G

        if data_prev is not None:
            # Concat previous labels/fake images to the ones before.
            prev_images = concat_frames(
                data_prev['prev_images'],
                net_G_output['fake_images'].detach(), num_frames_G - 1)

            # if 'alphas' in data:
            #     print("can't get gt segm from fake images, consider lowering max_sequence_length to num_frames_G")
            # if 'thetas' in data:
            #     print("can't get gt segm from fake images, consider lowering max_sequence_length to num_frames_G")
        else:
            assert t >= num_frames_G - 1
            prev_images = data['images'][:, t-(num_frames_G-1):t]

        data_t = dict()
        data_t['image'] = image
        data_t['prev_images'] = prev_images
        data_t['real_prev_image'] = data['images'][:, t - 1] if t > 0 else None
        if (self.use_stereo or self.use_stereo_detached or self.use_stereo_forward or self.use_stereo_ssi):# and data_prev is None:
            # assert data_prev is None
            if 'stereo_T' in data:
                data_t['stereo_T'] = data['stereo_T'][:, t]
                data_t[('K', 0)] = data[('K', 0)][:, t]
                # data_t[('inv_K', 0)] = data[('inv_K', 0)][:, t]

                data_t['stereo_image'] = data['stereo_images'][:, t]
                data_t['stereo_prev_images'] = data['stereo_images'][:, t-(num_frames_G-1):t]
                data_t['stereo_real_prev_image'] = data['stereo_images'][:, t - 1] if t > 0 else None
            elif 'src_w2c' in data:
                # previously used t instead of t-1 which was wrong
                # it doesn't matter for kitti because extrinsics in one clip remain the same across time
                data_t['src_K'] = data['src_K'][:, t-1]
                data_t['tgt_K'] = data['tgt_K'][:, t-1]
                data_t['src_w2c'] = data['src_w2c'][:, t-1]
                data_t['tgt_w2c'] = data['tgt_w2c'][:, t-1]
                # data_t['stereo_image'] = data['stereo_images'][:, t]
                data_t['stereo_prev_images'] = data['stereo_images'][:, t - (num_frames_G - 1):t]
                # data_t['bds'] = data['bds'][:, t-1]
                # data_t['fds'] = data['fds'][:, t-1]
                if 'depths' in data:
                    bs = data['images'].shape[0]
                    data_t['depths'] = [data['depths'][i][t-1] for i in range(bs)]
                    data_t['xys_cam'] = [data['xys_cam'][i][t-1] for i in range(bs)]
            else:
                # data_t['prev_cameras'] = data['cameras'][:, t-(num_frames_G-1):t]
                # data_t['prev_poses'] = data['poses'][:, t-(num_frames_G-1):t]
                data_t['query_image'] = data['query_image']
                data_t['query_camera'] = data['query_camera']
                data_t['query_pose'] = data['query_pose']
                # data_t['src_xyzs'] = data['src_xyzs']  # corresponds to I_t
                # data_t['prev_xyzs'] = data['xyzs'][:, t-(num_frames_G-1):t]
                data_t['src_camera'] = data['cameras'][:, t-1]
                data_t['src_pose'] = data['poses'][:, t-1]
                if self.cfg.data.name not in ['mvcam_colmap', 'mvcam_256_768_colmap', 'mvcam_256_colmap', 'mvcam_128_384_colmap']:
                    data_t['src_xyzs'] = [data['xyzs'][i][t-1] for i in range(data_t['image'].shape[0])]
                    data_t['src_xys_cam'] = [data['xys_cam'][i][t-1] for i in range(data_t['image'].shape[0])]
                # data_t['query_xyzs'] = data['query_xyzs']
                # data_t['query_xys_cam'] = data['query_xys_cam']
            if 'bds' in data:
                data_t['bds'] = data['bds'][:, t-1]
                data_t['fds'] = data['fds'][:, t-1]
        data_t['tp1'] = t
        return data_t

    def _init_tensorboard(self):
        super()._init_tensorboard()
        self.val_meters = dict()

    def write_metrics(self):
        # validation

        if self.val_data_loader is None:
            print('val data loader is None, do nothing')
            return

        # a hack so that self.gen_losses keeps its original state after evaluation
        # self.gen_losses should be only updated but not reassigned to a new dict
        # otherwise tensorboard logging will be incorrect
        train_gen_loss = copy.copy(self.gen_losses)
        for k in train_gen_loss.keys():
            del self.gen_losses[k]

        val_gen_loss = dict()

        with torch.no_grad():
            for data in tqdm(self.val_data_loader):
                data = to_cuda(data)

                # Get the data for the first t
                data_t = self.get_data_t(data, None, None, t=self.cfg.data.num_frames_G - 1)
                net_G_output = self.net_G(data_t, compute_loss=True)
                self._get_gen_losses(data_t=data_t,
                                     net_G_output=net_G_output,
                                     net_D_output={})

                for k, v in self.gen_losses.items():
                    if k not in val_gen_loss:
                        val_gen_loss[k] = []
                    val_gen_loss[k].append(v)

                # TODO: extra evaluation metrics as in evaluate.py

                if os.getenv('DEBUG') == '1':
                    print('=========')
                    print('XXX DEBUG, only run one batch for validation')
                    print('==========')
                    break

        for k, v in val_gen_loss.items():
            val_gen_loss[k] = torch.stack(val_gen_loss[k]).mean()

        self.gen_losses.update(train_gen_loss)
        self.val_gen_losses.update(val_gen_loss)

        # write to meters

        for k, v in val_gen_loss.items():
            full_name = f"val/{k}"
            if full_name not in self.val_meters.keys():
                # Create a new meter if it doesn't exist.
                self.val_meters[full_name] = Meter(full_name)
            self.val_meters[full_name].write(v.item())

        self._flush_meters(self.val_meters)

        if is_master():
            save_path = 'epoch_{:05}_iteration_{:09}_write_metrics.pt'.format(
                self.current_epoch, self.current_iteration)
            save_path = os.path.join(self.cfg.logdir, save_path)
            torch.save(
                {
                    'current_epoch': self.current_epoch,
                    'current_iteration': self.current_iteration,
                    'val_gen_loss': self.val_gen_losses,
                    'gen_loss': self.gen_losses,
                },
                save_path,
            )
            print('[INFO] save val_gen_loss, gen_loss to {}'.format(save_path))

    def save_image(self, path, data):
        r"""Save the output images to path.
        Note when the generate_raw_output is FALSE. Then,
        first_net_G_output['fake_raw_images'] is None and will not be displayed.
        In model average mode, we will plot the flow visualization twice.
        Args:
            path (str): Save path.
            data (dict): Training data for current iteration.
        """
        if not hasattr(self, 'batch_for_inference'):
            if self.val_data_loader is None:
                print('[ERROR] missing validation dataset')
                dataloader = self.train_data_loader
                dataset = self.train_data_loader.dataset
            else:
                dataloader = self.val_data_loader
                dataset = self.val_data_loader.dataset

            old_sequence_length = self.sequence_length
            seq_len = self.cfg.trainer.inference_sequence_length
            if seq_len != -1:
                self.sequence_length = seq_len
                dataset.set_sequence_length(self.sequence_length)

            # list of dicts
            batch_size = self.cfg.trainer.num_videos_to_visualize
            # dataset_size = len(dataset)
            # items = [dataset[i] for i in range(get_rank(), dataset_size, max(dataset_size // batch_size, 1))]
            # print('inference indices', list(range(get_rank(), dataset_size, max(dataset_size // batch_size, 1))))
            items = [dataset[i] for i in range(batch_size)]
            print('inference batch size', len(items))
            print({k: v.shape for k, v in items[0].items() if isinstance(v, torch.Tensor)})
            print({k: v.dtype for k, v in items[0].items() if isinstance(v, torch.Tensor)})
            data = []
            for i in range(batch_size):
                data_i = to_cuda(dataloader.collate_fn([items[i]]))
                data.append(data_i)
            # data = to_cuda(self.train_data_loader.collate_fn(items))
            self.batch_for_inference = data

            self.sequence_length = old_sequence_length
            dataset.set_sequence_length(self.sequence_length)

            # visualize from the training set
            dataset = self.train_dataset
            # dataset_size = len(dataset)
            # items = [dataset[i] for i in range(get_rank()+1, dataset_size, max(dataset_size // batch_size, 1))]
            items = [dataset[i] for i in range(batch_size, 2 * batch_size)]
            for i in range(batch_size):
                data_i = to_cuda(dataloader.collate_fn([items[i]]))
                data.append(data_i)
            print('inference batch size for training split', len(items), 'total', len(self.batch_for_inference))

        data = self.batch_for_inference

        self.net_G.eval()
        self.net_G_output = None

        all_info = []
        for i in range(len(data)):
            data_i = data[i]
            with torch.no_grad():
                _, _, all_info_i = self.gen_frames(data_i)
                all_info.append(all_info_i)

        self.display(data, all_info)

        self.net_G.float()

    def display(self, data, all_info, early_stop=True, visualizers=None, row_ind_suffix='', save_prefix=''):
        # if not is_master():
        #     return

        # setup visualizer
        if not hasattr(self, 'visualize_helper'):
            # to use in evaluate.py
            self.visualize_helper = VisualizeHelper(self.cfg)

        if visualizers is None:
            if not hasattr(self, 'visualizers'):
                self.visualizers = dict()

                # os.makedirs(os.path.join(self.cfg.logdir, 'all_epochs'), exist_ok=True)
                # for i in range(data['images'].shape[0]):
                #     self.visualizers[i] = HTMLTableVisualizer(os.path.join(self.cfg.logdir, 'all_epochs', f"batch_{i:03d}"), f"batch_{i:03d}", persist_row_counter=True)
                #     self.visualizers[i].begin_html()

            # overwritten each time when save_image is called
            self.visualizers['latest'] = HTMLTableVisualizer(self.cfg.logdir, 'latest', persist_row_counter=True)
            self.visualizers['latest'].begin_html()
            self.visualize_helper.print_url(self.visualizers['latest'])

            visualizers = self.visualizers

        epoch_info = 'epoch_{:05}_iteration_{:09}'.format(
            self.current_epoch, self.current_iteration)
        if row_ind_suffix:
            row_ind_suffix = '-' + row_ind_suffix
        if save_prefix:
            save_prefix += '_'

        # save stuff to {logdir}/batch_{i}/final.png
        # and under {logdir}/all_epochs/batch_{i}
        for i in range(len(all_info)):
            frames = self.visualize_helper.collect_frames(
                # data={k: v[i:i + 1] if isinstance(v, torch.Tensor) else v for k, v in data.items()},
                data=data[i],
                model_inputs=[{'prev_images': model_input_t['prev_images']} for model_input_t in
                              all_info[i]['inputs']],
                model_outputs=[{'fake_images': model_output_t['fake_images']} for model_output_t in
                               all_info[i]['outputs']],
                model_auxiliary=[{'gt_warp_out': model_output_t['auxiliary']['gt_warp_out']
                                 if 'gt_warp_out' in model_output_t['auxiliary']
                                 else torch.zeros_like(model_output_t['fake_images'])}
                                 for model_output_t in all_info[i]['outputs']],
            )
            if i in visualizers:
                self.visualize_helper.display_frames(
                    vis=visualizers[i],
                    frames=frames,
                    table_name=f"batch_{i:03d}_{save_prefix}{epoch_info}",
                )
            if 'latest' in visualizers:
                self.visualize_helper.display_frames(
                    vis=visualizers['latest'],
                    frames=frames,
                    table_name=f"batch_{i:03d}_{save_prefix}{epoch_info}",
                )

            for t, (model_input_t, model_output_t) in enumerate(zip(all_info[i]['inputs'], all_info[i]['outputs'])):
                # model_input_t_i = {k: v[i:i + 1] if isinstance(v, torch.Tensor) else v for k, v in
                #                    model_input_t.items()}
                # model_output_t_i = {k: v[i:i + 1] for k, v in model_output_t.items() if not isinstance(v, dict)}
                # model_auxiliary_t_i = {k: v[i:i + 1] for k, v in model_output_t['auxiliary'].items()}
                # name_to_image = self.visualize_helper.collect_tensors(model_input_t_i, model_output_t_i,
                #                                                       model_auxiliary_t_i)
                name_to_image = self.visualize_helper.collect_tensors(model_input_t, model_output_t, model_output_t['auxiliary'])
                if i in visualizers:
                    self.visualize_helper.display_tensors(vis=visualizers[i], name_to_image=name_to_image,
                                                          table_name=f"{save_prefix}{epoch_info}_t_{model_input_t['tp1']}",
                                                          row_ind_suffix=row_ind_suffix)
                if 'latest' in visualizers:
                    self.visualize_helper.display_tensors(vis=visualizers['latest'], name_to_image=name_to_image,
                                                          table_name=f"batch_{i:03d}_{save_prefix}{epoch_info}_t_{model_input_t['tp1']}",
                                                          row_ind_suffix=row_ind_suffix)

                if early_stop:
                    break

        if 'latest' in visualizers:
            visualizers['latest'].end_html()


class VisualizeHelper(BaseHTMLHelper):
    fake_pad = 0
    gt_pad = 0.8
    diff_pad = 0.4

    def __init__(self, cfg):
        self._denormalize_rgb = get_denormalize_rgb(cfg)
        self.mpi_front_depth, self.mpi_back_depth = cfg.gen.embed.front_depth, cfg.gen.embed.back_depth

    def collect_tensors(self, model_input_t, model_output_t, model_auxiliary_t):

        # constants
        # layout_np_pad = 255  # 242
        fake_pad = self.fake_pad
        gt_pad = self.gt_pad
        diff_pad = self.diff_pad
        denormalize_rgb = self._denormalize_rgb

        name_to_image = dict()  # a map from name to np arrays

        # transform from (-1, 1) or (0, 1) to (0, 1)
        fake_image = denormalize_rgb(model_output_t['fake_images'])#.clamp_(0, 1)
        real_image = denormalize_rgb(model_input_t['image'])

        name_to_image['fake_image'] = collect_tensor(fake_image, pad_value=fake_pad)
        name_to_image['real_image'] = collect_tensor(real_image, pad_value=gt_pad)
        name_to_image['real_minus_fake'] = collect_tensor((real_image - fake_image).abs(), pad_value=diff_pad)

        if 'fake_image_refined' in model_output_t:
            fake_image_refined = denormalize_rgb(model_output_t['fake_image_refined'])#.clamp_(0, 1)
            name_to_image['fake_image_r'] = collect_tensor(fake_image_refined, pad_value=fake_pad)
            name_to_image['real_minus_fake_r'] = collect_tensor((real_image - fake_image_refined).abs(), pad_value=diff_pad)

        if model_input_t['prev_images'].shape[1] > 0:
            prev_images = denormalize_rgb(model_input_t['prev_images'])
            name_to_image['prev_images'] = collect_tensor(prev_images, pad_value=self.gt_pad)

            prev_image = denormalize_rgb(model_input_t['prev_images'][:, -1])
            name_to_image['prev_image'] = collect_tensor(prev_image, pad_value=gt_pad)
            real_minus_prev = (real_image - prev_image).abs()
            name_to_image['real_minus_prev'] = collect_tensor(real_minus_prev, pad_value=diff_pad)

            # recon_prev_image = denormalize_rgb(model_auxiliary_t['recon_prev_image'])#.clamp_(0, 1)
            recon_prev_image = denormalize_rgb(model_output_t['recon_image'])
            name_to_image['recon_prev_image'] = collect_tensor(recon_prev_image, pad_value=fake_pad)
            name_to_image['prev_minus_recon'] = collect_tensor((prev_image - recon_prev_image).abs(), pad_value=diff_pad)
            name_to_image['fake_minus_fake_prev'] = collect_tensor((fake_image - recon_prev_image).abs(), pad_value=diff_pad)

            # normalize real_minus_prev and fake_minus_fake_prev
            fake_minus_fake_prev = (fake_image - recon_prev_image).abs()
            amax = max(real_minus_prev.max(), fake_minus_fake_prev.max())
            amin = min(real_minus_prev.min(), fake_minus_fake_prev.min())
            name_to_image['real_minus_prev_normalized'] = collect_tensor((real_minus_prev - amin) / (amax - amin), pad_value=diff_pad)
            name_to_image['fake_minus_fake_prev_normalized'] = collect_tensor((fake_minus_fake_prev - amin) / (amax - amin), pad_value=diff_pad)


        # for t, im in enumerate(prev_images):
        #     name_to_image[f"prev_images_{t}"] = collect_tensor(im, pad_value=gt_pad)

        # if 'background' in model_output_t:
        #     os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)
        #     background = self._denormalize_rgb(model_output_t['background'])
        #     imageio.imwrite(os.path.join(save_dir, 'raw', f"{save_prefix}_background.png"), collect_tensor(background))
        #
        warp_in = warp_out = None
        if 'proj_latent_rgba_layers' in model_output_t:
            warp_in = model_output_t['proj_latent_rgba_layers']  # proj(latent_rgba)
            warp_out = model_output_t['proj_warped_latent_rgba_layers']  # proj(warp(latent_rgba))
        elif 'rgba_layers' in model_output_t:
            warp_in = model_output_t['rgba_layers']
            warp_out = model_output_t['warped_rgba_layers']
        elif 'warped_rgba_layers' in model_output_t:
            warp_out = model_output_t['warped_rgba_layers']

        n_channels = model_output_t['fake_images'].shape[-3]
        if warp_in is not None:
            warp_in_color, warp_in_alpha = warp_in.split((n_channels, 1), dim=-3)
            warp_in_color = denormalize_rgb(warp_in_color)#.clamp_(0, 1)
            warp_in = torch.cat([warp_in_color, warp_in_alpha], dim=-3)

            name_to_image.update({f"warp_input_{k}": v for k, v in visualize_rgba(warp_in).items()})

        if warp_out is not None:
            warp_out_color, warp_out_alpha = warp_out.split((n_channels, 1), dim=-3)
            warp_out_color = denormalize_rgb(warp_out_color)#.clamp_(0, 1)
            warp_out = torch.cat([warp_out_color, warp_out_alpha], dim=-3)

            name_to_image.update({f"warp_output_{k}": v for k, v in visualize_rgba(warp_out).items()})

        if warp_in is not None and warp_out is not None:
            name_to_image['warp_color_layers_diff'] = collect_tensor((warp_out_color - warp_in_color).abs())

        if 'rgba_layers_refined' in model_output_t:
            warp_in_color_refined, warp_in_alpha_refined = model_output_t['rgba_layers_refined'].split((n_channels, 1), dim=-3)
            warp_in_color_refined = denormalize_rgb(warp_in_color_refined)
            warp_in_refined = torch.cat([warp_in_color_refined, warp_in_alpha_refined], dim=-3)

            name_to_image.update({f"warp_input_r_{k}": v for k, v in visualize_rgba(warp_in_refined).items()})

            disparity_refined = mpi.compute_disparity(warp_in_alpha_refined, front_depth=1, back_depth=10)
            name_to_image['disparity_r'] = collect_tensor(disparity_refined, pad_value=fake_pad)#.squeeze(-1)
            name_to_image['depth_r'] = collect_tensor(1 / disparity_refined / 10, pad_value=fake_pad)#.squeeze(-1)

            warp_out_color_refined, warp_out_alpha_refined = model_output_t['warped_rgba_layers_refined'].split((n_channels, 1), dim=-3)
            warp_out_color_refined = denormalize_rgb(warp_out_color_refined)
            warp_out_refined = torch.cat([warp_out_color_refined, warp_out_alpha_refined], dim=-3)

            name_to_image.update({f"warp_output_r_{k}": v for k, v in visualize_rgba(warp_out_refined).items()})

            if model_input_t['prev_images'].shape[1] > 0:
                prev_image = denormalize_rgb(model_input_t['prev_images'][:, -1])

                recon_prev_image_refined = denormalize_rgb(mpi.compose_back_to_front(model_output_t['rgba_layers_refined']))
                name_to_image['recon_prev_image_r'] = collect_tensor(recon_prev_image_refined, pad_value=fake_pad)
                name_to_image['prev_minus_recon_r'] = collect_tensor((prev_image - recon_prev_image_refined).abs(), pad_value=diff_pad)

        # # save theta as .txt
        # if 'theta_layers' in model_output_t:
        #     theta = model_output_t['theta_layers'].flatten(end_dim=-3)  # (..., 2, 3) -> (b, 2, 3)
        #     to_write = [f"index {i}\n{theta[i]}" for i in range(theta.shape[0])]
        #     to_write = '\n'.join([save_prefix, ''] + to_write)
        #     with open(os.path.join(save_dir, f'theta.txt'), 'a') as f:
        #         f.write(to_write)

        # # flow
        # flow_layers = None
        # if 'flow_layers' in model_output_t:
        #     flow_layers = model_output_t['flow_layers']
        # elif 'flow_layers' in model_auxiliary_t:
        norm_flow_layers = grid_to_norm_flow(model_output_t['coordinates'].flatten(0, 1)).unflatten(0, model_output_t['coordinates'].shape[:2])
        norm_flow = mpi.compose_back_to_front(torch.cat([norm_flow_layers, warp_in_alpha], dim=-3)).clamp(-1, 1)
        norm_flow_u = norm_flow[:, 0:1, :, :]
        norm_flow_v = norm_flow[:, 1:2, :, :]

        scale_u = norm_flow_u.abs().max()
        scale_v = norm_flow_v.abs().max()

        if 'gt_flow' in model_auxiliary_t:
            gt_norm_flow = flow_to_norm_flow(model_auxiliary_t['gt_flow']).clamp(-1, 1)
            gt_norm_flow_u = gt_norm_flow[:, 0:1, :, :]
            gt_norm_flow_v = gt_norm_flow[:, 1:2, :, :]

            scale_u = max(scale_u, gt_norm_flow_u.abs().max())
            scale_v = max(scale_v, gt_norm_flow_v.abs().max())

            gt_norm_flow_u = gt_norm_flow_u / scale_u
            gt_norm_flow_v = gt_norm_flow_v / scale_v

            name_to_image['gt_norm_flow_u'] = dict(image=collect_tensor(gt_norm_flow_u * 0.5 + 0.5), info=f"scale={scale_u}")
            name_to_image['gt_norm_flow_v'] = dict(image=collect_tensor(gt_norm_flow_v * 0.5 + 0.5), info=f"scale={scale_v}")

        norm_flow_u = norm_flow_u / scale_u
        norm_flow_v = norm_flow_v / scale_v

        name_to_image['norm_flow_u'] = dict(image=collect_tensor(norm_flow_u * 0.5 + 0.5), info=f"scale={scale_u}")
        name_to_image['norm_flow_v'] = dict(image=collect_tensor(norm_flow_v * 0.5 + 0.5), info=f"scale={scale_v}")

        # flow_layers = model_auxiliary_t['flow_layers']
        flow_layers = model_output_t['flow_layers']
        # elif 'coordinates' in model_output_t:
        #     flow_layers = grid_to_flow(model_output_t['coordinates'].flatten(0, 1)).unflatten(0, model_output_t['coordinates'].shape[:2])
        if flow_layers is not None and warp_in is not None:
            name_to_image['flow_layers'] = collect_tensor(flow_layers, process_grid=tensor2flow, value_check=False)
            fake_flow = mpi.compose_soft(colors=flow_layers, alphas=warp_in_alpha)
            fake_flow_conf = compute_flow_conf(fake_flow, im1=model_input_t['image'], im2=model_input_t['prev_images'][:, -1])
            name_to_image['flow'] = collect_tensor(fake_flow, process_grid=tensor2flow, value_check=False)
            name_to_image['flow_conf'] = collect_tensor(fake_flow_conf)#.squeeze(-1)
        # name_to_image_this = visualize_flow(flow_layers)
        # name_to_image.update({f"flow_layers_{k}": v for k, v in name_to_image_this.items()})

        if 'gt_flow' in model_auxiliary_t:
            name_to_image['gt_flow'] = collect_tensor(model_auxiliary_t['gt_flow'], process_grid=tensor2flow, value_check=False)
            name_to_image['gt_flow_conf'] = collect_tensor(model_auxiliary_t['gt_flow_conf'])#.squeeze(-1)
            # if 'gt_flow_conf1' in model_auxiliary_t:
            #     name_to_image['gt_flow_conf1'] = collect_tensor(model_auxiliary_t['gt_flow_conf1']).squeeze(-1)
            #     name_to_image['gt_flow_conf2'] = collect_tensor(model_auxiliary_t['gt_flow_conf2']).squeeze(-1)
            # cfg.logdir = os.path.join(ROOT_DIR, 'logs/debug')

            # if not os.path.exists(os.path.join(save_dir, 'gt_flow')):
            #     os.makedirs(os.path.join(save_dir, 'gt_flow'), exist_ok=True)
            #     imageio.imwrite(os.path.join(save_dir, 'gt_flow', f"gt_flow.png"), name_to_image['gt_flow'])
            #
            #     gt_warp_out = self._denormalize_rgb(model_auxiliary_t['gt_warp_out'])
            #     imageio.imwrite(os.path.join(save_dir, 'gt_flow', f"gt_warp_out.png"), collect_tensor(gt_warp_out))
            #     imageio.imwrite(os.path.join(save_dir, 'gt_flow', f"gt_warp_out_minus_prev.png"), collect_tensor((gt_warp_out - prev_image).abs()))
            #     imageio.imwrite(os.path.join(save_dir, 'gt_flow', f"real_minus_gt_warp_out.png"), collect_tensor((real_image - gt_warp_out).abs()))

        # "recenter" -> take max, min on the fly and recenter to range [0, 1]
        # "unscale" -> know that inputs in range [min, max] in advance and use this bound to unscale to [0, 1]
        # for monodepth2, this min = 0.1, max = 100 for both pesudo ground truth depth and mpi depth

        def tensor_to_unit_range(t):
            # normalize tensor to unit range [0, 1]
            # so that it's suitable for visualization (not all white or black)
            divider = t.max() - t.min()
            if divider < 1e-5:
                print('tensor is almost uniform')
                return torch.ones_like(t)
            return (t - t.min()) / divider

        if 'gt_disp' in model_auxiliary_t:
            gt_disp = model_auxiliary_t['gt_disp']
            name_to_image['gt_disp'] = collect_tensor(tensor_to_unit_range(gt_disp))

            # gt_disp_next = model_auxiliary_t['gt_disp_next']
            # name_to_image['gt_disp_next'] = collect_tensor(tensor_to_unit_range(gt_disp_next))

            # gt_depth = 1 / gt_disp.clamp(min=1e-5)
            # name_to_image['gt_depth'] = collect_tensor(normalize_disp(gt_depth)).squeeze(-1)

            # gt_unscaled_depth = model_auxiliary_t['gt_unscaled_depth']
            # name_to_image['gt_unscaled_depth'] = collect_tensor(gt_unscaled_depth)
            # gt_unscaled_depth_next = model_auxiliary_t['gt_unscaled_depth_next']
            # name_to_image['gt_unscaled_depth_next'] = collect_tensor(gt_unscaled_depth_next)

        # back_disp, front_disp = 1 / self.mpi_back_depth, 1 / self.mpi_front_depth
        # assert back_disp < front_disp
        disparity = model_output_t['pred_disparity']
        # name_to_image['unscaled_disparity'] = collect_tensor((disparity - back_disp) / (front_disp - back_disp))
        name_to_image['disparity'] = collect_tensor(tensor_to_unit_range(disparity))
        disparity_next = model_output_t['pred_disparity_next']
        # name_to_image['unscaled_disparity_next'] = collect_tensor((disparity_next - back_disp) / (front_disp - back_disp))
        name_to_image['disparity_next'] = collect_tensor(tensor_to_unit_range(disparity_next))

        if 'pred_disparity_ssi' in model_auxiliary_t:
            disparity_ssi = model_auxiliary_t['pred_disparity_ssi']
            name_to_image['disparity_ssi'] = collect_tensor(tensor_to_unit_range(disparity_ssi))

        # forward project from main view to stereo view
        target_stereo_view = None
        if 'stereo_prev_images' in model_input_t:
            target_stereo_view = model_input_t['stereo_prev_images'][:, -1]
        elif 'query_image' in model_input_t:
            target_stereo_view = model_input_t['query_image']  # it's not stereo
        if target_stereo_view is not None:
            target_stereo_view = denormalize_rgb(target_stereo_view)
            name_to_image['target_stereo_view'] = collect_tensor(target_stereo_view)

        syn_forward_stereo_view_mpi_inv_disp = None
        if False:#'syn_forward_stereo_view_mpi_inv_disp' in model_auxiliary_t:
            syn_forward_stereo_view_mpi_inv_disp = model_auxiliary_t['syn_forward_stereo_view_mpi_inv_disp']
            syn_tgt_rgba = model_auxiliary_t['syn_forward_stereo_view_before_compose_mpi_inv_disp']
        elif False:#'colmap_sparse_disp_map' in model_auxiliary_t:
            syn_forward_stereo_view_mpi_inv_disp = model_auxiliary_t['trg_rgb_syn']
            syn_tgt_rgba = model_auxiliary_t['trg_rgba_layers_syn']

            sparse_depth_map = model_auxiliary_t['colmap_sparse_depth_map']
            t = sparse_depth_map[sparse_depth_map > 0]
            sparse_depth_map[sparse_depth_map > 0] = (sparse_depth_map[sparse_depth_map > 0] - t.min()) / (t.max() - t.min())
            name_to_image['colmap_sparse_depth_map'] = collect_tensor(sparse_depth_map)
            sparse_disp_map = model_auxiliary_t['colmap_sparse_disp_map']
            t = sparse_disp_map[sparse_disp_map > 0]
            sparse_disp_map[sparse_disp_map > 0] = (sparse_disp_map[sparse_disp_map > 0] - t.min()) / (t.max() - t.min())

            name_to_image['colmap_sparse_disp_map'] = collect_tensor(sparse_disp_map)
        elif 'trg_rgba_t' in model_auxiliary_t and 'trg_rgba_tp1' not in model_auxiliary_t:
            syn_tgt_rgba = model_auxiliary_t['trg_rgba_t']
            syn_forward_stereo_view_mpi_inv_disp = model_auxiliary_t['trg_rgb_syn_t']

        elif False:#'trg_rgba_t' in model_auxiliary_t:
            syn_tgt_rgba = model_auxiliary_t['trg_rgba_t']
            syn_forward_stereo_view_mpi_inv_disp = model_auxiliary_t['trg_rgb_syn_t']

            trg_rgba = model_auxiliary_t['trg_rgba_tp1']
            trg_color, trg_alpha = trg_rgba.split((n_channels, 1), dim=-3)
            trg_color = denormalize_rgb(trg_color)
            trg_rgba = torch.cat([trg_color, trg_alpha], dim=-3)
            name_to_image.update({f"trg_tp1_{k}": v for k, v in visualize_rgba(trg_rgba).items()})

            trg_rgb_syn = model_auxiliary_t['trg_rgb_syn_tp1']
            trg_rgb_syn = denormalize_rgb(trg_rgb_syn)
            name_to_image['trg_rgb_syn_tp1'] = collect_tensor(trg_rgb_syn)

            trg_rgb = model_input_t['stereo_image']
            trg_rgb = denormalize_rgb(trg_rgb)
            name_to_image['trg_rgb_tp1'] = collect_tensor(trg_rgb)
            name_to_image['trg_rgb_tp1_error'] = collect_tensor((trg_rgb - trg_rgb_syn).abs())

        if syn_forward_stereo_view_mpi_inv_disp is not None:
            syn_forward_stereo_view_mpi_inv_disp = denormalize_rgb(syn_forward_stereo_view_mpi_inv_disp)
            name_to_image['syn_forward_stereo_view_mpi_inv_disp'] = collect_tensor(syn_forward_stereo_view_mpi_inv_disp)
            name_to_image['syn_forward_stereo_view_mpi_inv_disp_error'] = collect_tensor((syn_forward_stereo_view_mpi_inv_disp - target_stereo_view).abs())

            syn_tgt_color, syn_tgt_alpha = syn_tgt_rgba.split((n_channels, 1), dim=-3)
            syn_tgt_color = denormalize_rgb(syn_tgt_color)  # .clamp_(0, 1)
            syn_tgt_rgba = torch.cat([syn_tgt_color, syn_tgt_alpha], dim=-3)
            name_to_image.update({f"syn_tgt_{k}": v for k, v in visualize_rgba(syn_tgt_rgba).items()})
            name_to_image['syn_color_layers_diff'] = collect_tensor((warp_in_color - syn_tgt_color).abs())

        if 'colmap_sparse_disp_map' in model_auxiliary_t:
            sparse_disp_map = model_auxiliary_t['colmap_sparse_disp_map']
            t = sparse_disp_map[sparse_disp_map > 0]
            sparse_disp_map[sparse_disp_map > 0] = (t - t.min()) / (t.max() - t.min())
            name_to_image['colmap_sparse_disp_map'] = collect_tensor(sparse_disp_map)
        # TODO
        if False:
            syn_stereo_view_gt_depth = model_auxiliary_t['syn_stereo_view_gt_depth']
            syn_stereo_view_gt_depth = denormalize_rgb(syn_stereo_view_gt_depth)
            name_to_image['syn_stereo_view_gt_depth'] = collect_tensor(syn_stereo_view_gt_depth)
            name_to_image['syn_stereo_view_gt_depthh_error'] = collect_tensor((syn_stereo_view_gt_depth - target_stereo_view).abs())

        # # for mpi representation, depth is exactly the inverse of disparity
        # depth = model_output_t['pred_depth']
        # name_to_image['unscaled_depth'] = collect_tensor((depth - self.mpi_front_depth) / (self.mpi_back_depth - self.mpi_front_depth))
        # depth_next = model_output_t['pred_depth_next']
        # name_to_image['unscaled_depth_next'] = collect_tensor((depth_next - self.mpi_front_depth) / (self.mpi_back_depth - self.mpi_front_depth))

        # disparity = None
        # if 'disp' in model_auxiliary_t:
        #     disparity = model_auxiliary_t['disp']
        #     # print('pred disparity', disparity.max(), disparity.min())
        #     # disparity = normalize_disp(disparity)  # also normalize to compare with depth labels?
        # else:
        #     assert gt_disp is None
        #     alpha = None
        #     if warp_in is not None:
        #         alpha = warp_in_alpha
        #     elif warp_out is not None:
        #         alpha = warp_out_alpha
        #     if alpha is not None:
        #         disparity = mpi.compute_disparity(alpha, front_depth=1, back_depth=10)
        #
        # if disparity is not None:
        #     name_to_image['disparity'] = collect_tensor(disparity, pad_value=fake_pad)#.squeeze(-1)
        #     name_to_image['depth'] = collect_tensor(1 / disparity / 10, pad_value=fake_pad)#.squeeze(-1)
        #
        # if 'disp_next' in model_auxiliary_t:
        #     disparity_next = model_auxiliary_t['disp_next']
        #     name_to_image['disparity_next'] = collect_tensor(disparity_next, pad_value=fake_pad)#.squeeze(-1)
        #     name_to_image['depth_next'] = collect_tensor(1 / disparity_next / 10, pad_value=fake_pad)#.squeeze(-1)

        return name_to_image

        # if 'gt_flow' in model_auxiliary_t:
        #     gt_warp_out_diff_layouts = stack_layouts([
        #         [name_to_image['prev_image'], name_to_image['prev_image'], np.ones_like(name_to_image['prev_image']) * layout_np_pad],
        #         [name_to_image['real_image'], name_to_image['gt_warp_out'], name_to_image['real_minus_gt_warp_out']],
        #         [name_to_image['real_minus_prev'], name_to_image['gt_warp_out_minus_prev'], name_to_image['real_minus_gt_warp_out']],
        #     ], pad_value=layout_np_pad)
        #
        #     final_image_row.append(gt_warp_out_diff_layouts)
        #     final_image_row.append(name_to_image['gt_flow_flow_vs_ang_vs_mag'])

    @property
    def layouts(self):
        visualize_rgba_keys = ['refined_rgba_layers', 'rgba_layers', 'color_layers', 'alpha_layers', 'transmittance_layers']
        layouts = [
            ['.', 'prev_image', 'recon_prev_image', 'prev_minus_recon', 'gt_disp', 'disparity', 'disparity_ssi', 'colmap_sparse_disp_map', 'gt_flow', 'flow'] + [f"warp_input_{k}" for k in visualize_rgba_keys],
            [f'prev_images', 'real_image', 'fake_image', 'real_minus_fake', 'gt_disp_next', 'disparity_next', '.', 'colmap_sparse_depth_map', 'gt_flow_conf', 'flow_conf'] + [f"warp_output_{k}" for k in visualize_rgba_keys],
            ['.', 'real_minus_prev', 'fake_minus_fake_prev', '.', '.', '.', '.', '.', 'gt_norm_flow_u', 'norm_flow_u'] + ['.', '.', 'warp_color_layers_diff', '.', '.'],
            ['.', 'real_minus_prev_normalized', 'fake_minus_fake_prev_normalized', '.', '.', '.', '.', '.', 'gt_norm_flow_v', 'norm_flow_v'],
            ]

        layouts = [
            layouts[0],
            layouts[1] + ['flow_layers'],
            layouts[2],
            layouts[3],
            # ['.', 'syn_forward_stereo_view_mpi_inv_disp_scaled', 'syn_forward_stereo_view_mpi_inv_disp'],
            # ['.', 'target_stereo_view', 'target_stereo_view'],
            # ['.', 'syn_forward_stereo_view_mpi_inv_disp_scaled_error', 'syn_forward_stereo_view_mpi_inv_disp_error'],
            ['.', 'syn_forward_stereo_view_mpi_inv_disp', '.', '.', '.', '.', '.', '.', '.', '.'] + [f"syn_tgt_{k}" for k in visualize_rgba_keys],
            ['.', 'target_stereo_view'],
            ['.', 'syn_forward_stereo_view_mpi_inv_disp_error', '.', '.', '.', '.', '.', '.', '.', '.'] + ['.', '.', 'syn_color_layers_diff'],
            ['.', 'trg_rgb_syn_tp1', '.', '.', '.', '.', '.', '.', '.', '.'] + [f"trg_tp1_{k}" for k in visualize_rgba_keys],
            ['.', 'trg_rgb_tp1'],
            ['.', 'synthesis_tp1_error', '.', '.', '.', '.', '.', '.', '.', '.'] + ['.', '.', '.'],
            # ['.', '.', 'recon_prev_image_r', 'prev_minus_recon_r', '.', '.'] + ['.'] * 2 + [f'warp_input_r_{k}' for k in visualize_rgba_keys],
            # ['.', '.', 'fake_image_r', 'real_minus_fake_r', 'disparity_r', 'depth_r'] + ['.'] * 2 + [f'warp_output_r_{k}' for k in visualize_rgba_keys],
            ]

        n_cols = max(len(layout_row) for layout_row in layouts)
        layouts = [
            layout_row + ['.' for _ in range(n_cols - len(layout_row))]
            for layout_row in layouts
        ]
        return layouts

    def display_tensors(self, vis, name_to_image, table_name, row_ind_suffix=''):
        n_rows = len(self.layouts)
        n_cols = len(self.layouts[0])

        layout = []
        for row in range(n_rows):
            row_args = []
            for col in range(n_cols):
                name = self.layouts[row][col]
                if name_to_image.get(name) is not None:
                    if isinstance(name_to_image[name], dict):
                        image = Image.fromarray(name_to_image[name]['image'])
                        info = name + "::" + name_to_image[name]['info']
                    else:
                        image = Image.fromarray(name_to_image[name])
                        info = name
                    row_args.append(dict(image=image, info=info))
                else:
                    # if name not in ['.']:
                    #     if '_r_' not in name and not name.endswith('_r'):
                    #         print(name, 'not found')
                    row_args.append(None)
            layout.append(row_args)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.dump_table(vis=vis, layout=layout, table_name=table_name, col_type='image', row_ind_suffix=row_ind_suffix)

    def collect_frames(self, data, model_inputs, model_outputs, model_auxiliary):
        # save predicted video
        real_frames = [data['images'][:, t] for t in range(data['images'].shape[1])]

        pre_fake_frames = model_inputs[0]['prev_images']
        pre_fake_frames = [pre_fake_frames[:, t] for t in range(pre_fake_frames.shape[1])]

        fake_frames = pre_fake_frames + [model_output_t['fake_images'] for model_output_t in model_outputs]

        frames_gt_flow = pre_fake_frames + [model_auxiliary_t['gt_warp_out'] for model_auxiliary_t in model_auxiliary]

        frames = []
        assert len(real_frames) == len(fake_frames) == len(frames_gt_flow), (len(real_frames), len(fake_frames), len(frames_gt_flow))
        for real_frame, fake_frame, frame_gt_flow in zip(real_frames, fake_frames, frames_gt_flow):
            frame = stack_layouts(
                [[
                    collect_tensor(self._denormalize_rgb(real_frame), pad_value=self.gt_pad),
                    collect_tensor(self._denormalize_rgb(fake_frame), #.clamp_(0, 1),
                                   pad_value=self.fake_pad),
                    # collect_tensor(self._denormalize_rgb(real_frame), pad_value=self.gt_pad),
                    # collect_tensor(self._denormalize_rgb(frame_gt_flow), pad_value=self.gt_pad)
                ]]
            )
            frames.append(frame)

        return frames

    def display_frames(self, vis, frames, table_name, row_ind_suffix=''):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.dump_table(vis, layout=[[frames]], table_name=table_name, col_type='video', row_ind_suffix=row_ind_suffix)
