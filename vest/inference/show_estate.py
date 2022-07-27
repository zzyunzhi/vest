import ipdb
from vest.flows.grid_utils import grid_to_flow, compute_flow_conf, grid_to_norm_flow, flow_to_norm_flow

import torch.nn.functional as F
from vest.third_party.mine.visualizations.image_to_video import img_tensor_to_np, disparity_normalization_vis
from tu.configs import AttrDict
from functools import partial
from PIL import Image
from vest.utils import mpi as mpi
from vest.flows.grid_utils import flow_to_norm_flow
from tu.flow import tensor2flow
import numpy as np
import json
from tu.loggers.utils import collect_tensor
import warnings
import importlib
from tqdm import tqdm
import argparse
import os
import torch
from vest.utils.visualization.view_syn import eval_checkpoint, ViewSynthesisHelper, verbose_update_attr
from vest.utils.visualization.html_table import HTMLTableVisualizer
from vest.utils.distributed import master_only_print as print
from vest.utils.misc import to_cuda
from vest.utils.imlogging import get_date_uid
from vest.utils.trainer import set_random_seed
from tu.loggers.utils import print_viscam_url
from vest.utils.dataset import _get_data_loader
from vest.inference.utils import apply_crop_no_pad


helper = ViewSynthesisHelper()
global_info = {}
train_dataloader = None
test_dataloader = None

flowNet = None


def get_dataloader(cfg, is_test, batch_size=1, shuffle=True):
    global test_dataloader
    global train_dataloader
    if is_test and test_dataloader is not None:
        return test_dataloader
    if not is_test and train_dataloader is not None:
        return train_dataloader
    dataset_module = importlib.import_module(cfg.data.type if not is_test else cfg.test_data.type)
    ds = dataset_module.Dataset(cfg, is_inference=is_test, is_test=is_test)
    sampler = None
    if not shuffle:
        batch_start_ind = np.random.choice(len(ds) - batch_size)
        print('[INFO] setting batch_start_ind to', batch_start_ind)
        # sampler = torch.utils.data.sampler.SequentialSampler(list(range(batch_start_ind, len(ds))))
        sampler = list(range(batch_start_ind, len(ds)))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler,
        pin_memory=True, num_workers=0,#2 if os.getenv('DEBUG') != '1' else 0,
        collate_fn=importlib.import_module(cfg.data.collate_fn_type).collate_fn if hasattr(cfg.data, 'collate_fn_type') else None,
    )
    if is_test:
        test_dataloader = dl
    else:
        train_dataloader = dl
    return dl


def run(show_batch_fn, trainer, visualizer, dataloader, batch_size):
    dl = iter(dataloader)
    success_count = 0
    for batch_ind in tqdm(range(batch_size)):
        success = show_batch_fn(trainer, next(dl), visualizer, f'test_{batch_ind}')
        if success:
            success_count += 1
        # print(f'[INFO] better than baseline count {success_count} / {batch_ind + 1}')
    return {}


def turn_off_stereo(trainer):
    verbose_update_attr(trainer, 'use_stereo_forward', False)
    verbose_update_attr(trainer, 'use_stereo', False)
    verbose_update_attr(trainer, 'use_stereo_detached', False)
    verbose_update_attr(trainer.net_G_module, 'use_stereo_forward', False)
    verbose_update_attr(trainer.net_G_module, 'use_stereo', False)
    verbose_update_attr(trainer.net_G_module, 'use_stereo_detached', False)


def run_wrapper(run_fn, checkpoint_logdir, pred_length=1, seed=0, is_test=True, shuffle=True):
    global global_info
    is_initialized = False
    for cfg, model, checkpoint_info in eval_checkpoint(checkpoint_logdir, seed=seed):
        if not is_initialized:
            sequence_length = pred_length + cfg.data.num_frames_G - 1
            verbose_update_attr(cfg.data.train, 'initial_sequence_length', sequence_length)
            verbose_update_attr(cfg.data.train, 'max_sequence_length', sequence_length)

            layout = [
                [print_viscam_url(cfg.logdir)],
                [print_viscam_url(f"experiments/console_out/{cfg.slurm_job_id}_{cfg.slurm_job_name}.out")],
                [cfg.logdir],
            ]

            cfg.logdir = None

            print('[INFO] is_test', is_test)
            if not is_test and 'kitti' in cfg.data.name:
                verbose_update_attr(cfg.data.generator_params, 'do_flip_p', 0)
                verbose_update_attr(cfg.data.generator_params, 'do_color_aug_p', 0)
            if 'estate' in cfg.data.name:
                test_data_info = {
                    'split': 'videoautoencoder_test_720p',
                    'use_lmdb': True,
                    'cache_file': '/viscam/data/estate/metadata_estate_synsin_eval_videoautoencoder_videoautoencoder_test_720p_1636585423.8567386.pkl',
                    'lmdb_paths': [
                       '/viscam/data/estate/bundles_from_svl4/colmap_exhaustive_matcher_lmdb_compact',
                    ],
                    'rgb_lmdb_paths': [
                       '/viscam/data/estate/bundles_from_svl4/frames_w256_h256_lmdb_compact',
                    ],
                    'legacy_image_path': False,
                    'rgb_cache_file_height': 256,
                    'rgb_cache_file_width': 256,
                    # 'load_percentage': 0.01,
                    'sample_strategy': 'random_positive',
                    # 'sample_strategy': 'random_5',
                    # 'batch_size': 4,
                }
                test_data_info = {
                    'split': 'test_360p',
                    'use_lmdb': True,
                    'cache_file': '/viscam/data/estate/metadata_estate_synsin_eval_videoautoencoder_360p_test_360p_1636692947.8723748.pkl',
                    'lmdb_paths': [
                        '/svl/u/yzzhang/estate/test_360p/colmap_lmdb_compact/',
                    ],
                    'rgb_lmdb_paths': [
                        '/svl/u/yzzhang/estate/test_360p_w256_h256/frames_w256_h256_lmdb_compact/',
                    ],
                    'legacy_image_path': True,
                    'rgb_cache_file_height': 256,
                    'rgb_cache_file_width': 256,
                    # 'load_percentage': 0.005,
                    # 'sample_strategy': 'random_positive',
                    # 'sample_strategy': 'random_5',
                    # 'sample_strategy': 'random',
                    'sample_strategy': 'random_10',
                    # 'batch_size': 4,
                }
                test_data_info = {
                    'test': test_data_info,
                    # 'type': 'vest.datasets.estate_synsin_eval_v2',
                    # 'type': 'vest.datasets.estate_synsin_eval_test',
                    'type': 'vest.datasets.estate_synsin_eval_test_mine',
                    # 'type': 'vest.datasets.estate_synsin_eval_demo',
                }
                test_data_info = AttrDict(test_data_info)
                verbose_update_attr(cfg, 'test_data', test_data_info)
                global_info[('global', 'cfg.test_data')] = cfg.test_data
            if 'acid' in cfg.data.name:
                test_data_info = {
                    'split': 'test_360p',
                    'use_lmdb': True,
                    'cache_file': '/viscam/data/acid/metadata_acid_synsin_eval_lmdb_test_360p_1636591990.5726135.pkl',
                    'lmdb_paths': ['/viscam/data/acid/bundles_from_viscam4/test_360p/colmap_lmdb_compact/'],
                    'rgb_lmdb_paths': ['/viscam/data/acid/bundles_from_viscam4/test_360p_w256_h160/frames_w256_h160_lmdb_compact'],
                    'legacy_image_path': True,
                    'rgb_cache_file_height': 160,
                    'rgb_cache_file_width': 256,
                    # 'load_percentage': 1,
                    # 'batch_size': 8,
                    'sample_strategy': 'random_positive',
                }
                test_data_info_baseline50 = {
                    'split': 'baseline50_360p',
                    'use_lmdb': True,
                    'cache_file': '/viscam/data/acid/metadata_acid_synsin_eval_lmdb_baseline50_360p_1636804819.3946586.pkl',
                    'lmdb_paths': ['/viscam/data/acid/bundles_from_svl5/baseline50_360p/colmap_sequential_matcher_lmdb_compact/'],
                    'rgb_lmdb_paths': ['/viscam/data/acid/bundles_from_svl5/baseline50_360p/frames_w256_h160_lmdb_compact'],
                    'legacy_image_path': False,
                    'rgb_cache_file_height': 160,
                    'rgb_cache_file_width': 256,
                    # 'load_percentage': 1,
                    # 'batch_size': 8,
                    'sample_strategy': None,
                }
                test_data_info = {
                    'test': test_data_info,
                    # 'test': test_data_info_baseline50,
                    'type': 'vest.datasets.estate_synsin_eval_test_acid',
                    # 'type': 'vest.datasets.estate_synsin_eval_test_acid_extr_time',
                }
                test_data_info = AttrDict(test_data_info)
                verbose_update_attr(cfg, 'test_data', test_data_info)
                global_info[('global', 'cfg.test_data')] = cfg.test_data
            if 'cloud' in cfg.data.name or 'cater' in cfg.data.name:
                test_data_info = {}
                test_data_info = {
                    'test': test_data_info,
                    'type': 'vest.datasets.clouds'
                }
                test_data_info = AttrDict(test_data_info)
                verbose_update_attr(cfg, 'test_data', test_data_info)
                global_info[('global', 'cfg.test_data')] = cfg.test_data
            if 'mvcam' in cfg.data.name:
                test_data_info = {
                    'split': 'test',
                }
                test_data_info = {
                    'test': test_data_info,
                    'type': cfg.data.type,
                    # 'type': "vest.datasets.mvcam_colmap" if 'colmap' in cfg.data.name else "vest.datasets.mvcam"
                }
                test_data_info = AttrDict(test_data_info)
                # verbose_update_attr(cfg.data.generator_params, 'stride', 1)
                verbose_update_attr(cfg, 'test_data', test_data_info)
                global_info[('global', 'cfg.test_data')] = cfg.test_data

            dataloader = get_dataloader(cfg, is_test=is_test, shuffle=shuffle)

            is_initialized = True

        key = checkpoint_logdir, checkpoint_info['epoch'], checkpoint_info['iteration']
        if key in global_info:
            continue

        logdir = os.path.join('logs/evaluation', f"{cfg.data.name}_{get_date_uid()}_{cfg.slurm_job_id}")
        print(f'Make folder {logdir}')
        os.makedirs(logdir, exist_ok=True)
        visualizer = HTMLTableVisualizer(logdir, f"test", persist_row_counter=True)
        visualizer.begin_html()
        url = helper.print_url(visualizer, verbose=False)
        print('[URL] ' + url)
        global_info[key] = {'url': url}
        helper.dump_table(visualizer, layout, table_name='info', col_type='code')
        helper.dump_table(visualizer, [[
            f"epoch: {checkpoint_info['epoch']}, iter: {checkpoint_info['iteration']}, {checkpoint_info['checkpoint_path']}"
        ]], table_name='info', col_type='code')

        if not torch.cuda.is_available():
            # it's a hack since some tensorflow baseline uses environments with torch-cpu
            trainer = AttrDict(cfg=cfg)
        else:
            trainer_lib = importlib.import_module(cfg.trainer.type)
            trainer = trainer_lib.Trainer(cfg, model, None, None, None, None, None, None, None)
            trainer.sequence_length = sequence_length
            trainer.current_epoch, trainer.current_iteration = checkpoint_info['epoch'], checkpoint_info['iteration']

        info = run_fn(trainer=trainer, visualizer=visualizer, dataloader=dataloader)
        global_info[key].update(info)

        helper.dump_table(visualizer, layout=[[json.dumps({str(k): v for k, v in global_info.items()}, sort_keys=True, indent=4)]], table_name='global_info')

    visualizer.end_html()
    print(json.dumps({str(k): v for k, v in global_info.items()}, sort_keys=True, indent=4))


def render_png(image, background='checker'):
    height, width, _ = image.shape[-3:]
    if background == 'checker':
        checkerboard = np.kron([[136, 120] * (width//128+1), [120, 136] * (width//128+1)] * (height//128+1), np.ones((16, 16)))
        checkerboard = np.expand_dims(np.tile(checkerboard, (4, 4)), -1)
        bg = checkerboard[:height, :width]
    elif background == 'black':
        bg = np.zeros([height, width, 1])
    elif background == 'white':
        bg = 255 * np.ones([height, width, 1])
    image = image.astype(np.float32)
    alpha = image[..., :, :, 3:] / 255
    rendered_image = alpha * image[..., :, :, :3] + (1 - alpha) * bg
    return rendered_image.astype(np.uint8)


def show_batch(trainer, data, visualizer, table_name='default'):
    do_turn_off_stereo = False#data['images'].shape[1] > 3 and 'kitti' not in trainer.cfg.data.name
    if do_turn_off_stereo:
        print('[INFO] found image shape', data['images'].shape)
        # FIXME: this is a hack
        turn_off_stereo(trainer)
    show_extr_time_video = True
    show_extr_time_images = True
    save_extr_time_images = False#False#True
    show_novel_view = False
    show_extr_space_video = False
    show_extr_space_images = False#True
    save_extr_space_images = False#False#True
    show_rgba_layers = False
    show_warped_rgba_layers = False
    show_frames = False
    show_maps = True
    show_frames_resized_no_crop = False# True
    show_flow_layers = False
    show_rgb_transmittance_layers = False
    do_resize = False
    if True and 'estate' in trainer.cfg.data.name:
        do_resize = True

        def apply_resize(x):
            return F.interpolate(x, size=(256, 448), mode='area')
    if 'mvcam' in trainer.cfg.data.name:
        do_resize = True

        def apply_resize(x):
            return F.interpolate(x, size=(256, 448), mode='area')
            # return F.interpolate(x, size=(144, 256), mode='bilinear', align_corners=False)

    do_crop = False

    data = to_cuda(data)
    with torch.no_grad():
        _, _, all_info = trainer.gen_frames(data)

    inputs = all_info['inputs'][0]
    outputs = all_info['outputs'][0] | all_info['outputs'][0]['auxiliary']
    if 'kitti' not in trainer.cfg.data.name and 'cloud' not in trainer.cfg.data.name and 'cater' not in trainer.cfg.data.name:
        assert 'scale_factor' in outputs, outputs.keys()

    real_frames = [data['images'][:, t] for t in range(data['images'].shape[1])]
    pre_fake_frames = inputs['prev_images']
    pre_fake_frames = [pre_fake_frames[:, t] for t in range(pre_fake_frames.shape[1])]
    fake_frames = pre_fake_frames + [outputs_t['fake_images'] for outputs_t in all_info['outputs']]
    layout = [[]]
    for frames in [real_frames, fake_frames]:
        frames = map(lambda x: x * 0.5 + 0.5, frames)
        if do_resize:
            frames = map(apply_resize, frames)
        if do_crop:
            frames = map(apply_crop_no_pad, frames)
        frames = map(collect_tensor, frames)
        frames = list(frames)
        layout[-1].append(frames)
    if show_extr_time_video:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            helper.dump_table(visualizer, layout, col_type='video', table_name=f"{table_name}_extr_time_l{len(real_frames)}")
    if show_extr_time_images:
        for frames in layout[-1]:
            helper.dump_table(visualizer, [[Image.fromarray(frame) for frame in frames]], table_name='extr_time_real_then_fake')
    if save_extr_time_images:
        for frame_ind, frame in enumerate(frames):
            path = os.path.join(visualizer.visdir,
                                f"extr_time_{table_name.removeprefix('test_')}t{frame_ind}.png")
            Image.fromarray(frame).save(path)
            if frame_ind == 0:
                print('[INFO] saved images to', os.path.abspath(path))
    if show_novel_view:
        syn_image = outputs['trg_rgb_syn_t'] * 0.5 + 0.5
        tgt_image = inputs['query_image'] * 0.5 + 0.5
        if do_resize:
            syn_image = apply_resize(syn_image)
            tgt_image = apply_resize(tgt_image)
        helper.dump_table(visualizer,
                          [[Image.fromarray(collect_tensor(tgt_image)),
                            Image.fromarray(collect_tensor(syn_image))]],
                          table_name='extr_space_real_then_fake')

    if False:
        layout = [[]]
        mpi_frames = [outputs_t['rgba_layers'] for outputs_t in all_info['outputs']]
        mpi_frames = [torch.cat([x[..., :-1, :, :] * 0.5 + 0.5, x[..., -1:, :, :]], dim=-3) for x in mpi_frames]
        for layer_ind in range(mpi_frames[0].shape[1]):
            frames = [x[:, layer_ind] for x in mpi_frames]
            frames = [apply_crop_no_pad(x) for x in frames]
            frames = [collect_tensor(x) for x in frames]
            frames = [render_png(x, background='white') for x in frames]
            layout[-1].append(frames)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            helper.dump_table(visualizer, layout, col_type='video', table_name=f"{table_name}_extr_time_l{len(real_frames)}")

    color, alpha = outputs['color_layers'], outputs['alpha_layers']
    color = color * 0.5 + 0.5
    rgba = torch.cat([color, alpha], dim=-3)
    if do_resize:
        rgba = apply_resize(rgba.flatten(0, 1))
        rgba = rgba.unflatten(0, (color.shape[0], rgba.shape[0] // color.shape[0]))
    if 'src_K' in inputs:
        h, w = color.shape[-2:]
        intrinsics = inputs['src_K']
        intrinsics = torch.stack([intrinsics[:, 0, 0] / w, intrinsics[:, 1, 1] / h,
                                  intrinsics[:, 0, 2] / w, intrinsics[:, 1, 2] / h], dim=-1).cuda()
    elif ('K', 0) in inputs:
        intrinsics = inputs[('K', 0)]
    else:
        h, w = color.shape[-2:]
        intrinsics = torch.Tensor([1, 1 * w / h, 0.5, 0.5]).cuda().view(1, 4)

    disparity = outputs['disparity_linspace']
    if 'scale_factor' in outputs:
        disparity /= outputs['scale_factor'].view(1, 1)
    depths = torch.reciprocal(disparity)

    if 'render_configs' not in global_info:
        if 'kitti' in trainer.cfg.data.name:
            render_configs = dataset_to_render_configs['kitti']
        elif 'estate' in trainer.cfg.data.name:
            if do_turn_off_stereo:
                render_configs = []
            else:
                render_configs = dataset_to_render_configs['estate']
        elif 'acid' in trainer.cfg.data.name:
            render_configs = dataset_to_render_configs['acid']
        elif 'cloud' in trainer.cfg.data.name:
            render_configs = []
        elif 'cater' in trainer.cfg.data.name:
            render_configs = []
        elif 'mvcam' in trainer.cfg.data.name:
            render_configs = dataset_to_render_configs['mvcam']
        global_info['render_configs'] = render_configs

    render_configs = global_info['render_configs']

    if show_extr_space_video:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            helper.render_sway(rgba, vis=visualizer, depths=depths, intrinsics=intrinsics,
                               # configs=render_configs,
                               configs=[render_configs[0]],
                               apply_crop_fn=apply_crop_no_pad if do_crop else None,
                               table_name=f"{table_name}_extr_space")
    if show_extr_space_images or save_extr_space_images:
        list_frames = helper.render_sway(rgba, vis=visualizer, depths=depths, intrinsics=intrinsics, configs=render_configs,
                                       apply_crop_fn=apply_crop_no_pad if do_crop else None,
                                       return_frames=True)
        if show_extr_space_images:
            helper.dump_table(visualizer, [[Image.fromarray(frame) for frame in frames]])

        if save_extr_space_images:
            for config_ind, frames in enumerate(list_frames):
                for frame_ind, frame in enumerate(frames):
                    path = os.path.join(visualizer.visdir, f"extr_time_{table_name.removeprefix('test_')}_config_{config_ind}_space_{frame_ind}.png")
                    Image.fromarray(frame).save(path)
                    if frame_ind == 0:
                        print('[INFO] saved images to', os.path.abspath(path))

    keys = []
    if show_rgba_layers:
        keys.append('rgba_layers')
    if show_warped_rgba_layers:
        keys.append('warped_rgba_layers')
    for key in keys:
        color, alpha = outputs[key].split((3, 1), dim=-3)
        color = color * 0.5 + 0.5
        rgba = torch.cat([color, alpha], dim=-3)

        layers = [rgba[:, i] for i in range(rgba.shape[1])]
        if do_resize:
            layers = map(apply_resize, layers)
        layers = map(collect_tensor, layers)
        layers = map(Image.fromarray, layers)
        layout = [list(layers)]
        helper.dump_table(visualizer, layout, table_name=key)

    if show_rgb_transmittance_layers:
        color, alpha = outputs['color_layers'] , outputs['alpha_layers']
        rgba = torch.cat([color * 0.5 + 0.5, mpi.layer_weights(alpha)], dim=-3)
        layout = [[Image.fromarray(collect_tensor(rgba[:, i])) for i in range(rgba.shape[1])]]
        helper.dump_table(visualizer, layout, table_name='rgb_transmittance')

    if show_flow_layers:
        flow = outputs['flow_layers']
        alpha = outputs['alpha_layers']
        layout = [[]]
        for i in range(alpha.shape[1]):
            flow_vis = collect_tensor(flow[:, i], process_grid=tensor2flow, value_check=False)
            alpha_vis = collect_tensor(alpha[:, i])[..., 0:1]
            layout[0].append(Image.fromarray(np.concatenate([flow_vis, alpha_vis], axis=-1)))
        helper.dump_table(visualizer, layout, table_name='flow-alpha layers')

    if show_frames_resized_no_crop:
        images = {}
        images['11'] = inputs['prev_images'][:, 0]
        images['21'] = inputs['prev_images'][:, 1]
        images['31'] = inputs['image']
        images['31_syn'] = outputs['fake_images']
        if 'stereo_prev_images' in inputs:
            images['22'] = inputs['stereo_prev_images'][:, -1]
        else:
            images['22'] = inputs['query_image']
        images['22_syn'] = outputs['trg_rgb_syn_t']
        for k, v in images.items():
            images[k] = v * 0.5 + 0.5
        images['31_syn_21_diff'] = (images['31_syn'] - images['21']).abs()
        images['21_11_diff'] = (images['21'] - images['11']).abs()
        images['22_syn_21_diff'] = (images['22_syn'] - images['21']).abs()
        # normalize diff
        for k, v in images.items():
            if k.endswith('diff'):
                images[k] = (v - v.min()) / (v.max() - v.min())
        # images['flow'] = outputs['pred_flow']
        # images['disp'] = outputs['pred_disparity']
        for k, v in images.items():
            if do_resize:
                v = apply_resize(v)
            if v.shape[-3] == 1:
                # depth map
                v = disparity_normalization_vis(v)
                v = img_tensor_to_np(v)
            elif v.shape[-3] == 2:
                # flow map
                v = collect_tensor(v, process_grid=tensor2flow, value_check=False)
            else:
                v = collect_tensor(v)
            images[k] = Image.fromarray(v)
        layout = [[images['11'], images['21']]]
        helper.dump_table(visualizer, layout, table_name="I11, I21")
        layout = [[images['31'], images['31_syn'], images['22'], images['22_syn']]]
        helper.dump_table(visualizer, layout, table_name="I31, fakeI31, I22, fakeI22")
        layout = [[images['21_11_diff'], images['31_syn_21_diff'], images['22_syn_21_diff']]]
        helper.dump_table(visualizer, layout, table_name='I31 diff, I22 diff')
        # layout = [[images['flow'], images['disp']]]
        # helper.dump_table(visualizer, layout, table_name="flow, disp")

    if show_frames:
        layout = [[collect_tensor(inputs['prev_images'][:, i] * 0.5 + 0.5) for i in range(inputs['prev_images'].shape[1])]]
        for i in range(len(layout[0])):
            layout[0][i] = Image.fromarray(layout[0][i])
        helper.dump_table(visualizer, layout, table_name="I_1, I_2")

        # I_3, hat_I_3, I_2', hat_I_2'
        layout = [[collect_tensor(inputs['image'] * 0.5 + 0.5),
                   collect_tensor(outputs['fake_images'] * 0.5 + 0.5),
                   collect_tensor(inputs['stereo_prev_images'][:, -1] * 0.5 + 0.5),
                   collect_tensor(outputs['trg_rgb_syn_t'] * 0.5 + 0.5)]]
        for i in range(len(layout[0])):
            layout[0][i] = Image.fromarray(layout[0][i])
        helper.dump_table(visualizer, layout, table_name="I_3, hat_I_3, I_2', hat_I_2'")

    if False:
        disparity = disparity.view(1, -1, 1, 1, 1).expand_as(alpha)
        disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            helper.render_sway(torch.cat([disparity, alpha], dim=-3), vis=visualizer,
                               depths=depths, intrinsics=intrinsics,
                               apply_crop_fn=apply_crop_no_pad,
                               table_name=f"{table_name}_extr_space")

    if show_maps:
        global flowNet
        if False and flowNet is None:
            # py39 for titanrtx, cu110 for 3090
            from vest.third_party.flow_net.flow_net import FlowNet
            flowNet = FlowNet(pretrained=True, fp16=True, rgb_max=1)

        norm_flow_layers = grid_to_norm_flow(outputs['coordinates'].flatten(0, 1)).unflatten(0, outputs['coordinates'].shape[:2])
        flow_is_valid = torch.logical_and(norm_flow_layers.gt(-0.6), norm_flow_layers.lt(0.6)).flatten(start_dim=2).all(dim=2)  # (b, d)
        if not flow_is_valid.all():
            message = f'[ERROR] some layers have unbounded flow: {flow_is_valid}'
            helper.dump_table(visualizer, [[message]], table_name=table_name, col_type='code')
        for i in range(len(flow_is_valid)):
            norm_flow_layers[i, ~flow_is_valid[i], :, :, :] = 0
        norm_flow = mpi.compose_back_to_front(torch.cat([norm_flow_layers, outputs['alpha_layers']], dim=-3)).clamp(-1, 1)
        if False:
            flow = outputs['pred_flow']
        else:
            flow = norm_flow
        # flow = apply_crop_no_pad(flow)
        if do_resize:
            flow = apply_resize(flow)
        flow = collect_tensor(flow, process_grid=tensor2flow, value_check=False)
        if False:#True:
            # compute ground truth flow
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flow_gt, flow_conf_gt = flowNet(
                    inputs['image'],
                    inputs['prev_images'][:, -1],
                )
            # flow_gt = apply_crop_no_pad(flow_gt)
            flow_gt = collect_tensor(flow_gt, process_grid=tensor2flow, value_check=False)
            # flow_conf_gt = apply_crop_no_pad(flow_conf_gt)
            # flow_conf_gt = collect_tensor(flow_conf_gt)
        disp = outputs['pred_disparity']
        if do_resize:
            disp = apply_resize(disp)
        # disp = apply_crop_no_pad(disp)
        # disp = collect_tensor(disp)
        disp = disparity_normalization_vis(disp)
        disp = img_tensor_to_np(disp)
        # layout = [[flow, flow_gt, flow_conf_gt, disp]]
        layout = [[flow, disp]]
        # layout = [[flow, flow_gt, disp]]
        for i in range(len(layout[0])):
            layout[0][i] = Image.fromarray(layout[0][i])
        helper.dump_table(visualizer, layout, table_name=table_name)

    if False:
        trainer.display([data], [all_info], visualizers={0: visualizer}, early_stop=False)

    return True


def main():
    # cater 3516805 3516806 3525320
    # cloud 3525320
    global global_info
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('path', nargs='*', help='Dir for loading models.')
    parser.add_argument('-d', '--dataset', help='kitti | estate')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed.')  # TODO: still not deterministic
    parser.add_argument('-ib', '--inference_batch_size', type=int, default=2, help='number of examples to display')
    parser.add_argument('-l', '--pred_length', type=int, default=1)
    parser.add_argument('--is_train', action='store_true', help='use train split instead of test split')
    parser.add_argument('--no_shuffle', action='store_true', help='shuffle dataloader')
    parser.add_argument('--best_only', action='store_true', help='use best.pt')
    args = parser.parse_args()

    set_random_seed(args.seed, by_rank=True)
    print(vars(args))
    global_info['args'] = vars(args)

    if args.dataset is not None:
        checkpoints = [dataset_to_checkpoint[args.dataset]]
    else:
        checkpoints = args.path
    run_fn = partial(run, show_batch_fn=show_batch, batch_size=args.inference_batch_size)
    for checkpoint in checkpoints:
        if args.best_only:
            checkpoint = os.path.join(checkpoint, 'best.pt')
        run_wrapper(run_fn, checkpoint, pred_length=args.pred_length, seed=args.seed,
                    is_test=not args.is_train, shuffle=not args.no_shuffle)

    print(json.dumps({str(k): v for k, v in global_info.items()}, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
