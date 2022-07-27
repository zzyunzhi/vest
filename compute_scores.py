from vest.inference.show_estate import run_wrapper, helper, global_info, turn_off_stereo
from functools import partial
from vest.utils.visualization.view_syn import verbose_update_attr
from tqdm import tqdm
from vest.utils.trainer import set_random_seed
import argparse
import json
import numpy as np
import torch.nn.functional as F
from vest.losses.similarity import apply_crop, SSIMMetric, PNSR, LPIPSAlex, LPIPSVGG, LPIPSVGGMINE, LPIPSAlexMINE, LPIPSVGGVideoAutoencoder
import torch.nn as nn
import os
import torch
from vest.utils.distributed import master_only_print as print
from vest.utils.misc import to_cuda
from vest.inference.utils import dump_kde
from vest.inference.utils import get_layout_row, process_render_results, get_render_results


similarity_metrics = None


def turn_off_sparse_matching(trainer):
    verbose_update_attr(trainer.net_G_module.stereo_loss, 'use_disp_scale', False)
    verbose_update_attr(trainer.net_G_module.stereo_loss, 'use_disp_scale_detached', False)


def run(test_batch_fn, trainer, visualizer, dataloader, logging_iter=None, quick=False):
    metrics = {}
    count_fail = 0
    count_succ = 0
    assert dataloader.batch_size == 1
    for data in tqdm(dataloader):
        metrics_this = test_batch_fn(trainer, data)
        if metrics_this is None:
            count_fail += 1
            print(f'[ERROR] skip batch, fail {count_fail} : succ {count_succ}')
            continue
        count_succ += 1
        for k, v in metrics_this.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v.item())
        if os.getenv('DEBUG') == '1' and len(metrics[k]) > 10:
            print('[DEBUG] only use 100 batches')
            break
        if logging_iter is not None and len(metrics[k]) % logging_iter == 0:
            print(json.dumps({k: np.mean(v) for k, v in metrics.items()}, sort_keys=True, indent=4))
        if quick and len(metrics[k]) >= 300:
            print('[INFO] only test on 100 samples, percentage', len(metrics[k]) / len(dataloader))
            break

    dump_kde(visualizer=visualizer, metrics=metrics)

    for k, v in metrics.items():
        metrics[k] = np.mean(v)

    metrics['counts'] = {'success': count_succ, 'failure': count_fail}

    print(json.dumps(metrics, sort_keys=True, indent=4))

    layout = [
        [json.dumps(metrics, sort_keys=True, indent=4)],
    ]
    helper.dump_table(visualizer, layout, table_name='scores' if os.getenv('DEBUG') != '1' else 'scores for 100 examples', col_type='code')
    return metrics


def test_batch(trainer, data, input_single_frame=False):
    global similarity_metrics
    if similarity_metrics is None:
        if 'estate' in trainer.cfg.data.name or 'mvcam' in trainer.cfg.data.name:
            similarity_metrics = {
                'ssim': SSIMMetric(),
                'pnsr': PNSR(),
                'lpips_vgg': LPIPSVGG(),
            }
        if 'acid' in trainer.cfg.data.name:
            similarity_metrics = {
                'mse': nn.MSELoss(),
                'lpips_vgg': LPIPSVGG(),
            }
        if 'kitti' in trainer.cfg.data.name:
            similarity_metrics = {
                'lpips_vgg_mine': LPIPSVGGMINE(),
                'lpips_vgg': LPIPSVGG(),
                'lpips_alex': LPIPSAlex(),
                'ssim': SSIMMetric(),
                'pnsr': PNSR(),
                # 'msssim': MSSSIM(),
            }
        for k, v in similarity_metrics.items():
            v.cuda()

    data = to_cuda(data)
    if input_single_frame:
        if 'stereo_T' in data:
            data = {
                'images': torch.stack([data['images'][:, 1],
                                       data['images'][:, 1],
                                       torch.zeros_like(data['images'][:, 1])], dim=1),
                'stereo_images': torch.stack([data['stereo_images'][:, 1],
                                              data['stereo_images'][:, 1],
                                              torch.zeros_like(data['stereo_images'][:, 1])], dim=1),
                'stereo_T': data['stereo_T'],
                ('K', 0): data[('K', 0)],
            }
        else:
            new_data = {}
            for k in ['images', 'cameras', 'poses']:
                new_data[k] = torch.stack([data[k][:, 1] for _ in range(3)], dim=1)
            for k in ['xyzs', 'xys_cam']:
                new_data[k] = [[data[k][i][1] for _ in range(3)] for i in range(len(data[k]))]
            for k in ['query_image', 'query_camera', 'query_pose']:
                new_data[k] = data[k]
            data = new_data
    if 'kitti' not in trainer.cfg.data.name and data['images'].shape[1] > 3 and trainer.use_stereo_forward:
        # hard-coded, ignore view synthesis evaluation
        turn_off_stereo(trainer)
    with torch.no_grad():
        _, _, all_info = trainer.gen_frames(data)

    inputs = all_info['inputs'][0]
    outputs = all_info['outputs'][0] | all_info['outputs'][0]['auxiliary']

    metrics_this = dict()

    """ scores for time """

    def update_for_t(t):
        real = all_info['inputs'][t-1]['image']
        fake = all_info['outputs'][t-1]['fake_images']
        real = real * 0.5 + 0.5
        fake = fake * 0.5 + 0.5
        suffix = f'_p{t}'

        for k, v in similarity_metrics.items():
            score_this = v(fake, real).mean()
            if torch.isnan(score_this) or torch.isinf(score_this).any():
                print('[ERROR] nan', k, v)
                score_this = torch.zeros(())
            metrics_this[f'eval_pred{suffix}_{k}'] = score_this

    assert len(all_info['outputs']) in [1, 5, 9]
    update_for_t(1)
    if len(all_info['outputs']) == 5:
        update_for_t(3)
        update_for_t(5)
    if len(all_info['outputs']) == 9:
        for t in range(2, 10):
            update_for_t(t)
        for k in similarity_metrics.keys():
            metrics_this[f'eval_pred_p9_average_{k}'] = np.mean([
                metrics_this[f'eval_pred_p{t}_{k}'].item() for t in range(1, 10)
            ])

    if 'trg_rgb_syn_t' not in outputs:
        return metrics_this

    """ scores for space """

    # DEBUG
    if False:
        render_results = get_render_results(inputs, outputs)
        _, scores = get_layout_row(render_results, do_resize=(128, 384), do_crop=True, similarity_metrics=similarity_metrics)
        for k, v in scores.items():
            metrics_this[f'eval_syn_ds_{k}_debug'] = np.array(v)

    if 'stereo_prev_images' in inputs:
        trg_syn_image = inputs['stereo_prev_images'][:, -1]
    else:
        trg_syn_image = inputs['query_image']
    syn_image = outputs['trg_rgb_syn_t']
    trg_syn_image = trg_syn_image * 0.5 + 0.5
    syn_image = syn_image * 0.5 + 0.5

    if 'kitti' in trainer.cfg.data.name:
        # MPI synthesis evaluation crops 5% from the boundary
        trg_syn_image = apply_crop(trg_syn_image)
        syn_image = apply_crop(syn_image)

    if trainer.cfg.data.name == 'kitti_city':
        # do extra evaluation on low res for kitti
        trg_syn_image_ds = F.interpolate(trg_syn_image, size=(128, 384), mode='bilinear', align_corners=False)
        syn_image_ds = F.interpolate(syn_image, size=(128, 384), mode='bilinear', align_corners=False)

    with torch.no_grad():
        for k, v in similarity_metrics.items():
            metrics_this[f'eval_syn_{k}'] = v(syn_image, trg_syn_image).mean()

            if trainer.cfg.data.name == 'kitti_city':
            # if 'kitti' in trainer.cfg.data.name:
                metrics_this[f'eval_syn_ds_{k}'] = v(syn_image_ds, trg_syn_image_ds).mean()

    return metrics_this


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-p', '--path', required=True, type=str, help='Dir for loading models.')
    parser.add_argument('-d', '--dataset', help='kitti | estate')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed.')  # TODO: still not deterministic
    parser.add_argument('-l', '--pred_length', type=int, default=1)
    parser.add_argument('--input_single_frame', action='store_true', help='duplicate I11 as inputs')
    parser.add_argument('--logging_iter', type=int, default=200)
    parser.add_argument('-q', '--quick', action='store_true', help='load less samples if set to true')
    parser.add_argument('--best_only', action='store_true', help='use best.pt')
    args = parser.parse_args()

    set_random_seed(args.seed, by_rank=True)
    print(vars(args))

    test_batch_fn = partial(test_batch, input_single_frame=args.input_single_frame)
    run_fn = partial(run, logging_iter=args.logging_iter, quick=args.quick, test_batch_fn=test_batch_fn)
    if args.best_only:
        checkpoint = os.path.join(args.path, 'best.pt')
    else:
        checkpoint = args.path
    run_wrapper(run_fn, checkpoint, pred_length=args.pred_length, seed=args.seed)

    print(json.dumps({str(k): v for k, v in global_info.items()}, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
