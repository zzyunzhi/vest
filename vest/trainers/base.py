import torch
import os
from vest.utils.distributed import master_only_print as print


class BaseTrainer(object):
    def load_checkpoint(self, cfg, checkpoint_path, resume=None, strict=True):
        r"""Load network weights, optimizer parameters, scheduler parameters
        from a checkpoint.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
            resume (bool or None): If not ``None``, will determine whether or
                not to load optimizers in addition to network weights.
        """
        if os.path.exists(checkpoint_path):
            # If checkpoint_path exists, we will load its weights to
            # initialize our network.
            if resume is None:
                resume = False
        elif os.path.exists(os.path.join(cfg.logdir, 'latest_checkpoint.txt')):
            # This is for resuming the training from the previously saved
            # checkpoint.
            fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
            with open(fn, 'r') as f:
                line = f.read().splitlines()
            checkpoint_path = os.path.join(cfg.logdir, line[0].split(' ')[-1])
            if resume is None:
                resume = True
        else:
            # checkpoint not found and not specified. We will train
            # everything from scratch.
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            return current_epoch, current_iteration
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        current_epoch = 0
        current_iteration = 0
        if resume:
            self.net_G.load_state_dict(checkpoint['net_G'])
            if not self.is_inference:
                self.net_D.load_state_dict(checkpoint['net_D'])
                if 'opt_G' in checkpoint:
                    self.opt_G.load_state_dict(checkpoint['opt_G'])
                    self.opt_D.load_state_dict(checkpoint['opt_D'])
                    self.sch_G.load_state_dict(checkpoint['sch_G'])
                    self.sch_D.load_state_dict(checkpoint['sch_D'])
                    current_epoch = checkpoint['current_epoch']
                    current_iteration = checkpoint['current_iteration']
                    print('Load from: {}'.format(checkpoint_path))
                else:
                    print('Load network weights only.')
        else:
            self.net_G.load_state_dict(checkpoint['net_G'], strict=strict)
            print('Load generator weights only.')
            current_epoch = checkpoint['current_epoch']
            current_iteration = checkpoint['current_iteration']

        print('Done with loading the checkpoint.')
        return current_epoch, current_iteration
