import torch
import torch.nn as nn
from vest.losses.perceptual import PerceptualLoss
from vest.utils.distributed import master_only_print as print
from vest.utils.meters import Meter
from vest.utils.misc import get_nested_attr, split_labels, to_cuda


class Trainer:
    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        # Initialize models and data loaders.
        self.cfg = cfg
        self.net_G = net_G
        # One wrapper (DDP)
        self.net_G_module = self.net_G.module
        self.val_data_loader = val_data_loader
        self.is_inference = train_data_loader is None
        self.opt_G = opt_G
        self.sch_G = sch_G
        self.train_data_loader = train_data_loader

        # Initialize loss functions.
        # All loss names have weights. Some have criterion modules.
        # Mapping from loss names to criterion modules.
        self.criteria = nn.ModuleDict()
        # Mapping from loss names to loss weights.
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())
        self.gen_losses = self.losses['gen_update']
        self.dis_losses = self.losses['dis_update']
        self._init_loss(cfg)
        for loss_name, loss_weight in self.weights.items():
            if loss_weight > 0:
                print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                    self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

        if self.is_inference:
            # The initialization steps below can be skipped during inference.
            return

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self._init_tensorboard()

        self.sequence_length = 1
        if not self.is_inference:
            self.train_dataset = self.train_data_loader.dataset
            self.sequence_length_max = \
                min(getattr(cfg.data.train, 'max_sequence_length', 100),
                    self.train_dataset.sequence_length_max)
        self.Tensor = torch.cuda.FloatTensor

        self.net_G_output = self.data_prev = None
        self.net_G_module = self.net_G.module
    def save_checkpoint(self, current_epoch, current_iteration):
        r"""Save network weights, optimizer parameters, scheduler parameters
        to a checkpoint.
        """
        _save_checkpoint(self.cfg,
                         self.net_G,
                         self.opt_G,
                         self.sch_G,
                         current_epoch, current_iteration)

    def start_of_epoch(self, current_epoch):
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch

    def start_of_iteration(self, data, current_iteration):
        data = to_cuda(data)
        self.current_iteration = current_iteration
        self.net_G.train()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        # Logging.
        self._end_of_iteration(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_iteration >= self.cfg.snapshot_save_start_iter and \
                current_iteration % self.cfg.snapshot_save_iter == 0:
            self.save_image(None, None)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics()
        # Compute image to be saved.
        elif current_iteration % self.cfg.image_save_iter == 0:
            self.save_image(None, None)
        elif current_iteration % self.cfg.image_display_iter == 0:
            self.save_image(None, None)
        if current_iteration % self.cfg.logging_iter == 0:
            self._write_tensorboard()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        if current_epoch >= self.cfg.snapshot_save_start_epoch and \
                current_epoch % self.cfg.snapshot_save_epoch == 0:
            self.save_image(None, None)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics()

    @staticmethod
    def _write_to_meters(data, meters):
        r"""Write values to meters."""
        for key, value in data.items():
            meters[key].write(value)
    def _write_loss_meters(self):
        r"""Write all loss values to tensorboard."""
        for update, losses in self.losses.items():
            # update is 'gen_update' or 'dis_update'.
            assert update == 'gen_update' or update == 'dis_update'
            for loss_name, loss in losses.items():
                full_loss_name = update + '/' + loss_name
                if full_loss_name not in self.meters.keys():
                    # Create a new meter if it doesn't exist.
                    self.meters[full_loss_name] = Meter(full_loss_name)
                self.meters[full_loss_name].write(loss.item())
    def _flush_meters(self, meters):
        r"""Flush all meters using the current iteration."""
        for meter in meters.values():
            meter.flush(self.current_iteration)
    def _write_tensorboard(self):
        self._write_to_meters({
                               'optim/gen_lr': self.sch_G.get_last_lr()[0],
                               },
                              self.meters)
        self._write_loss_meters()
        self._flush_meters(self.meters)
    def _assign_criteria(self, name, criterion, weight):
        r"""Assign training loss terms.

        Args:
            name (str): Loss name
            criterion (obj): Loss object.
            weight (float): Loss weight. It should be non-negative.
        """
        self.criteria[name] = criterion
        self.weights[name] = weight

    def _init_loss(self, cfg):
        r"""Initialize training loss terms. In vid2vid, in addition to
        the GAN loss, feature matching loss, and perceptual loss used in
        pix2pixHD, we also add temporal GAN (and feature matching) loss,
        and flow warping loss. Optionally, we can also add an additional
        face discriminator for the face region.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria = dict()
        self.weights = dict()
        trainer_cfg = cfg.trainer
        loss_weight = cfg.trainer.loss_weight

        # # GAN loss and feature matching loss.
        # self._assign_criteria('GAN',
        #                       GANLoss(trainer_cfg.gan_mode),
        #                       loss_weight.gan)
        # self._assign_criteria('FeatureMatching',
        #                       FeatureMatchingLoss(),
        #                       loss_weight.feature_matching)

        # Perceptual loss.
        perceptual_loss = cfg.trainer.perceptual_loss
        self._assign_criteria('Perceptual',
                              PerceptualLoss(
                                  cfg=cfg,
                                  network=perceptual_loss.mode,
                                  layers=perceptual_loss.layers,
                                  weights=perceptual_loss.weights,
                                  num_scales=getattr(perceptual_loss,
                                                     'num_scales', 1)),
                              loss_weight.perceptual)

        # L1 Loss.
        if getattr(loss_weight, 'L1', 0) > 0:
            self._assign_criteria('L1', torch.nn.L1Loss(), loss_weight.L1)

        # Whether to add an additional discriminator for specific regions.
        self.add_dis_cfg = getattr(self.cfg.dis, 'additional_discriminators', None)
        assert self.add_dis_cfg is None
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                add_dis_cfg = self.add_dis_cfg[name]
                self.weights['GAN_' + name] = add_dis_cfg.loss_weight
                self.weights['FeatureMatching_' + name] = \
                    loss_weight.feature_matching

        # Temporal GAN loss.
        self.num_temporal_scales = get_nested_attr(self.cfg.dis,
                                                   'temporal.num_scales', 0)
        for s in range(self.num_temporal_scales):
            self.weights['GAN_T%d' % s] = loss_weight.temporal_gan
            self.weights['FeatureMatching_T%d' % s] = \
                loss_weight.feature_matching

        # Flow loss. It consists of three parts: L1 loss compared to GT,
        # warping loss when used to warp images, and loss on the occlusion mask.
        self.use_flow = hasattr(cfg.gen, 'flow')
        assert not self.use_flow

        # Other custom losses.
        self._define_custom_losses()

    def _define_custom_losses(self):
        r"""All other custom losses are defined here."""
        pass

    def _start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch. When current_epoch is smaller than
        $(single_frame_epoch), we only train a single frame and the generator is
        just an image generator. After that, we start doing temporal training
        and train multiple frames. We will double the number of training frames
        every $(num_epochs_temporal_step) epochs.

        Args:
            current_epoch (int): Current number of epoch.
        """
        cfg = self.cfg
        # Only generates one frame at the beginning of training
        if current_epoch < cfg.single_frame_epoch:
            self.train_dataset.sequence_length = 1
        # Then add the temporal network to generator, and train multiple frames.
        elif current_epoch == cfg.single_frame_epoch:
            self.init_temporal_network()

        # Double the length of training sequence every few epochs.
        temp_epoch = current_epoch - cfg.single_frame_epoch
        if temp_epoch > 0:
            sequence_length = \
                cfg.data.train.initial_sequence_length * \
                (2 ** (temp_epoch // cfg.num_epochs_temporal_step))
            sequence_length = min(sequence_length, self.sequence_length_max)
            if sequence_length > self.sequence_length:
                self.sequence_length = sequence_length
                self.train_dataset.set_sequence_length(sequence_length)
                print('------- Updating sequence length to %d -------' %
                      sequence_length)

    def init_temporal_network(self):
        r"""Initialize temporal training when beginning to train multiple
        frames. Set the sequence length to $(initial_sequence_length).
        """
        self.tensorboard_init = False
        # Update training sequence length.
        self.sequence_length = self.cfg.data.train.initial_sequence_length
        if not self.is_inference:
            self.train_dataset.set_sequence_length(self.sequence_length)
            print('------ Now start training %d frames -------' %
                  self.sequence_length)

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Print the errors to console."""
        if not torch.distributed.is_initialized():
            if current_iteration % self.cfg.logging_iter == 0:
                message = '(epoch: %d, iters: %d) ' % (current_epoch,
                                                       current_iteration)
                for k, v in self.gen_losses.items():
                    if k != 'total':
                        message += '%s: %.3f,  ' % (k, v)
                message += '\n'
                for k, v in self.dis_losses.items():
                    if k != 'total':
                        message += '%s: %.3f,  ' % (k, v)
                print(message)

    def _init_tensorboard(self):
        r"""Initialize the tensorboard. For the SPADE model, we will record
        regular and FID, which is the average FID.
        """
        self.regular_fid_meter = Meter('FID/regular')
        if self.cfg.trainer.model_average:
            self.average_fid_meter = Meter('FID/average')
        self.image_meter = Meter('images')
        self.meters = {}
        names = ['optim/gen_lr'
                 ]
        for name in names:
            self.meters[name] = Meter(name)

from vest.utils.distributed import is_master, master_only
import os
@master_only
def _save_checkpoint(cfg,
                     net_G,
                     opt_G,
                     sch_G,
                     current_epoch, current_iteration):
    latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint.pt'.format(
        current_epoch, current_iteration)
    save_path = os.path.join(cfg.logdir, latest_checkpoint_path)
    torch.save(
        {
            'net_G': net_G.state_dict(),
            'opt_G': opt_G.state_dict(),
            'sch_G': sch_G.state_dict(),
            'current_epoch': current_epoch,
            'current_iteration': current_iteration,
        },
        save_path,
    )
    fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
    with open(fn, 'wt') as f:
        f.write('latest_checkpoint: %s' % latest_checkpoint_path)
    print('Save checkpoint to {}'.format(save_path))
    return save_path
