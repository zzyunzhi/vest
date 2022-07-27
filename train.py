from typing import Dict
import torch.backends.cudnn as cudnn
import argparse
import os
import json
from common import ROOT
import copy
from vest.config import recursive_update, Config, get_attr_dict, get_config_recursive, AttrDict
from vest.utils.dataset import get_train_and_val_dataloader
from vest.utils.distributed import init_dist
from vest.utils.distributed import master_only_print as print
from vest.utils.gpu_affinity import set_affinity
from vest.utils.imlogging import make_logging_dir
from vest.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)
from tu.ddp import is_master, master_only
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config', help='Path to the training config file.')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='')
    parser.add_argument('--single_gpu', action='store_true', help='disable ddp')
    args = parser.parse_args()
    return args


def get_config_from_json(config: str) -> Config:
    with open(config, 'r') as f:
        cfg = json.load(f)
    cfg = AttrDict(cfg)
    return cfg


def get_config_from_yaml(config: str, dataset: str) -> Config:
    cfg = get_config_recursive(config)

    data_cfg_path = os.path.join(ROOT, f'configs/datasets/{dataset}.yaml')
    recursive_update(cfg, get_attr_dict(data_cfg_path))

    # turn off amp
    cfg.gen_opt.fused_opt = False
    cfg.dis_opt.fused_opt = False

    assert not hasattr(cfg, 'initial_sequence_length')
    assert not hasattr(cfg, 'max_sequence_length')
    assert not hasattr(cfg, 'num_frames_G')
    # clipping num_downsamples
    max_num_downsamples = 0
    while (
            cfg.data.img_size[0] % 2 ** (max_num_downsamples + 1) == 0 and
            cfg.data.img_size[1] % 2 ** (max_num_downsamples + 1) == 0
    ):
        max_num_downsamples += 1
    if max_num_downsamples < cfg.gen.embed.num_downsamples:
        print('img size', cfg.data.img_size)
        print('maximum num_downsamples allowed', max_num_downsamples)
        print('clipping unet num_downsamples from', cfg.gen.embed.num_downsamples)

        cfg.gen.embed.num_downsamples = max_num_downsamples

    return cfg


def main():
    args = parse_args()

    if not args.single_gpu:
        local_rank = int(os.environ['LOCAL_RANK'])
        print('ddp local rank', local_rank)
        init_dist(local_rank)
        set_affinity(local_rank)
    else:
        print({k: v for k, v in vars(args).items() if v is not None})
        local_rank = 0

    set_random_seed(0, by_rank=True)

    if args.config.endswith('.json'):
        cfg = get_config_from_json(args.config)
    else:
        cfg = get_config_from_yaml(args.config, args.dataset)

    cfg.local_rank = local_rank

    cfg.logdir = os.path.join(ROOT, 'logs', cfg.data.name)
    if not is_master():
        cfg.logdir += f'_rank_{cfg.local_rank}'

    # master only
    make_logging_dir(cfg.logdir)
    # make logdir for all ranks
    os.makedirs(cfg.logdir, exist_ok=True)

    if is_master():
        # store cfg as json file
        with open(os.path.join(cfg.logdir, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, sort_keys=True, indent=4)

    cudnn.deterministic = False
    cudnn.benchmark = True
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, need_val_data_loader=hasattr(cfg.data, 'val'))
    net_G, _, opt_G, _, sch_G, _ = get_model_optimizer_and_scheduler(cfg, seed=0)
    trainer = get_trainer(cfg, net_G, None, opt_G, None, sch_G, None, train_data_loader, val_data_loader)
    current_epoch, current_iteration = 0, 0

    best_checkpoint_file = os.path.join(cfg.logdir, 'best.pt')
    best_val_gen_loss = None

    @master_only
    def best_save(**kwargs):
        print('===========================================================')
        if 'total' in trainer.val_gen_losses:
            print(f"val total loss = {trainer.val_gen_losses['total'].item()}, "
                  f"train total loss = {trainer.gen_losses['total'].item()}, "
                  f"epoch = {current_epoch}, iter = {current_iteration}")
        else:
            print(f"train total loss = {trainer.gen_losses['total'].item()}, "
                  f"epoch = {current_epoch}, iter = {current_iteration}")

        print(f"saving to {best_checkpoint_file}")
        print('===========================================================')

        torch.save(
            {
                'net_G': net_G.state_dict(),
                'opt_G': opt_G.state_dict(),
                'sch_G': sch_G.state_dict(),
                'current_epoch': current_epoch,
                'current_iteration': current_iteration,
                'gen_loss': {k: v.item() for k, v in trainer.gen_losses.items()},
                'val_gen_loss': {k: v.item() for k, v in trainer.val_gen_losses.items()},
                **kwargs,
            },
            best_checkpoint_file,
        )

    def get_loss_to_compare(loss: Dict[str, torch.Tensor]):
        key = 'eval_lpips_vgg_syn_forward_stereo_view_mpi_inv_disp'
        if key in loss:
            return loss[key]
        key = 'eval_lpips_alex_pred_from_mpi'
        if key in loss:
            return loss[key]
        return 0

    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_data_loader):
            data = trainer.start_of_iteration(data, current_iteration)
            trainer.gen_update(data)
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)

            if current_iteration >= cfg.max_iter:
                print('Done with training!!!')
                return
            break  # DEBUG
        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)

        if current_epoch % cfg.snapshot_save_epoch == 0:
            if best_val_gen_loss is None or get_loss_to_compare(trainer.val_gen_losses) < get_loss_to_compare(best_val_gen_loss):
                best_val_gen_loss = copy.copy(trainer.val_gen_losses)
                best_save()

    print('Done with training!!!')
    return


if __name__ == "__main__":
    main()
