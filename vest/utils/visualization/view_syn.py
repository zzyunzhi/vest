import copy

import torch
from scipy.interpolate import interp1d
import numpy as np
import json
from PIL import Image
from vest.utils.visualization.html_table import HTMLTableColumnDesc, HTMLTableVisualizer
import vest.utils.mpi as mpi
import os
from vest.utils.visualization.html_helper import BaseHTMLHelper
from vest.config import AttrDict, get_attr_dict
from tu.ddp import master_only_print
import imageio
from vest.trainers.base import BaseTrainer
import base64
from vest.utils.trainer import get_model_optimizer_and_scheduler
from common import ROOT
import io
from tu.loggers.utils import collect_tensor
from vest.utils.dataset import get_train_and_val_dataloader, get_test_dataloader, _get_test_dataset_object, _get_data_loader
import glob


def get_batch(cfg, is_test, sequence_length=None, batch_size=None):
    # cfg.data.name = os.path.basename(os.path.dirname(cfg.logdir))  # for old versions, data.name is incorrect
    print(cfg.data.name)

    if cfg.data.name.startswith('clevrer'):
        cfg.data.load_percentage = 0.01  # hacky

    if batch_size is not None:
        cfg.data.train.batch_size = batch_size
    if sequence_length is None:
        sequence_length = cfg.data.num_frames_G
    if sequence_length > cfg.data.train.max_sequence_length:
        print('required sequence length is larger than dataset configureation')
        cfg.data.train.max_sequence_length = sequence_length
    if is_test:
        test_dataset = _get_test_dataset_object(cfg)
        not_distributed = getattr(
            cfg.test_data, 'val_data_loader_not_distributed', False)
        not_distributed = 'video' in cfg.test_data.type or not_distributed
        test_data_loader = _get_data_loader(
            cfg, test_dataset, batch_size or cfg.test_data.test.batch_size, not_distributed,
            shuffle=True)
        train_data_loader = test_data_loader
    else:
        train_data_loader, _ = get_train_and_val_dataloader(cfg, need_val_data_loader=False)

    train_data_loader.dataset.set_sequence_length(sequence_length)

    data = next(iter(train_data_loader))

    return data


def get_data_t(cfg, sequence_length, batch_size):
    data = get_batch(cfg, sequence_length=sequence_length, batch_size=batch_size)

    t = cfg.data.num_frames_G - 1
    data_t = dict(image=data['images'][:, t], prev_images=data['images'][:, t - (cfg.data.num_frames_G - 1):t])
    if 'stereo_T' in data:
        data_t['stereo_T'] = data['stereo_T'][:, t]
        data_t[('K', 0)] = data[('K', 0)][:, t]
        data_t[('inv_K', 0)] = data[('inv_K', 0)][:, t]

        data_t['stereo_prev_images'] = data['stereo_images'][:, t - (cfg.data.num_frames_G - 1):t]

    return data_t


def verbose_setattr(c, key, value):
    if not hasattr(c, key):
        master_only_print(f"setting {key} to {value} which did not exist")
        setattr(c, key, value)


def verbose_delattr(c, key):
    master_only_print(f"deleting attribute", key, getattr(c, key))
    delattr(c, key)


def verbose_update_attr(c, key, value):
    master_only_print(f"overwritting {key} from {getattr(c, key)} to {value}")
    setattr(c, key, value)


def update_legacy_config(cfg):

    verbose_setattr(cfg.gen.embed, 'input_flow', False)
    verbose_setattr(cfg.gen.embed, 'input_eulerian', False)
    verbose_setattr(cfg.gen.embed, 'output_no_color', False)
    verbose_setattr(cfg.gen.embed, 'front_depth', 1)
    verbose_setattr(cfg.gen.embed, 'back_depth', 10)
    verbose_setattr(cfg.gen.embed, 'use_gradient_checkpoint', False)
    verbose_setattr(cfg.gen.embed, 'use_affine_motion', True)
    verbose_setattr(cfg.gen, 'use_pwc_net', False)
    verbose_setattr(cfg.gen, 'use_eulerian_motion', False)
    verbose_setattr(cfg.gen, 'use_mono', False)
    verbose_setattr(cfg.gen, 'use_midas', False)
    verbose_setattr(cfg.gen, 'use_monodepth2', False)
    verbose_setattr(cfg.gen, 'use_stereo', False)
    verbose_setattr(cfg.gen, 'use_stereo_detached', False)
    verbose_setattr(cfg.gen, 'use_stereo_forward', False)
    verbose_setattr(cfg.gen, 'use_stereo_ssi', False)
    verbose_setattr(cfg.gen, 'use_disp_scale', False)
    verbose_setattr(cfg.gen, 'use_disp_scale_detached', False)
    verbose_setattr(cfg.gen, 'use_pts_transformer', False)
    verbose_setattr(cfg.gen, 'no_flow_confidence', False)
    verbose_setattr(cfg.gen, 'flow_confidence_threshold', 0.02)

    verbose_setattr(cfg, "slurm_job_id", "dummy")
    verbose_setattr(cfg, "slurm_job_name", "dummy")

    if 'imaginaire' in cfg.trainer.type:
        cfg.trainer.type = cfg.trainer.type.replace('imaginaire', 'vest')
    if 'imaginaire' in cfg.gen.type:
        cfg.gen.type = cfg.gen.type.replace('imaginaire', 'vest')
    if 'imaginaire' in cfg.gen.embed.type:
        cfg.gen.embed.type = cfg.gen.embed.type.replace('imaginaire', 'vest')
    if 'imaginaire' in cfg.dis.type:
        cfg.dis.type = cfg.dis.type.replace('imaginaire', 'vest')
    if 'imaginaire' in cfg.data.type:
        cfg.data.type = cfg.data.type.replace('imaginaire', 'vest')
    if 'imaginaire' in cfg.test_data.type:
        cfg.test_data.type = cfg.test_data.type.replace('imaginaire', 'vest')

    if cfg.dis.type == 'vest.discriminators.dummy' and cfg.trainer.loss_weight.gan > 0:
        print('dummy discriminator')
        verbose_update_attr(cfg.trainer.loss_weight, 'gan', 0)
        verbose_update_attr(cfg.trainer.loss_weight, 'feature_matching', 0)

    if not os.path.exists(cfg.logdir):
        cfg.logdir = os.path.join(ROOT, cfg.logdir)

    data_single_sequence_path = os.path.join(cfg.logdir, 'data_single_sequence.pkl')
    if os.path.exists(data_single_sequence_path) and not hasattr(cfg, 'saved_data_single_sequence'):
        verbose_setattr(cfg, 'saved_data_single_sequence', data_single_sequence_path)

    if "kitti" in cfg.data.name:
        verbose_setattr(cfg.data.generator_params, "interpolation", "LANCZOS")
        verbose_setattr(cfg.data.generator_params, "dataset_variant", "raw_city")
        verbose_setattr(cfg.data.generator_params, "do_flip_p", 0)
        verbose_setattr(cfg.data.generator_params, "do_color_aug_p", 0)
        verbose_setattr(cfg.data.generator_params, "use_zero_tm1", False)
        verbose_update_attr(cfg.data.generator_params, "use_monodepth2_calibration", False)  # always load from cam files

    if cfg.data.name == 'mvcam':
        verbose_setattr(cfg.data.generator_params, 'use_neighbors', False)
    if 'acid' in cfg.data.name:
        verbose_setattr(cfg.data, 'dataset_name', 'acid')
    elif 'estate' in cfg.data.name:
        verbose_setattr(cfg.data, 'dataset_name', 'estate')
    # if cfg.data.name == 'estate_synsin_eval':
    #     verbose_update_attr(cfg.test_data, "type", "vest.datasets.estate_synsin_eval")
    #     verbose_setattr(cfg.test_data, "collate_fn_type", "vest.datasets.estate_synsin_eval")
    elif 'mvcam' in cfg.data.name:
        verbose_setattr(cfg.data.generator_params, "move_src_cam", False)
        verbose_setattr(cfg.data.generator_params, "colmap_root", '/viscam/data/mvcam/run_colmap_deep3d')


def eval_checkpoint(checkpoint_logdir, seed=0, strict=True, validation_task=None, validation_key=None):
    checkpoint_logdir = os.path.abspath(checkpoint_logdir)
    if os.path.splitext(checkpoint_logdir)[1] == '.pt':
        checkpoint_paths = [checkpoint_logdir]
        checkpoint_logdir = os.path.dirname(checkpoint_logdir)
    else:
        checkpoint_paths = list(sorted(glob.glob(f"{checkpoint_logdir}/epoch_*_iteration_*_checkpoint.pt")))
        if not checkpoint_paths:
            print('no checkpoint found', checkpoint_logdir)
            for _ in []:
                yield None, None, None
            return

        if validation_task is not None and validation_key is not None:
            print('[INFO] find the best checkpoint on validation split', validation_task, validation_key)
            get_loss_to_compare = {
                ('pred', 'lpips_vgg'): lambda loss: loss['eval_lpips_vgg_pred_from_mpi'],
                ('pred', 'lpips_alex'): lambda loss: loss['eval_lpips_alex_pred_from_mpi'],
                ('syn', 'lpips_vgg'): lambda loss: loss['eval_lpips_vgg_syn_forward_stereo_view_mpi_inv_disp'],
                ('syn', 'lpips_alex'): lambda loss: loss['eval_lpips_alex_syn_forward_stereo_view_mpi_inv_disp'],
            }[(validation_task, validation_key)]

            best_val_gen_loss = None
            best_checkpoint_path = None
            for checkpoint_path in checkpoint_paths:
                checkpoint = torch.load(
                    checkpoint_path.replace('_checkpoint.pt', '_write_metrics.pt'), map_location=lambda storage, loc: storage)
                val_gen_loss = checkpoint['val_gen_loss']
                if best_val_gen_loss is None or get_loss_to_compare(val_gen_loss) < get_loss_to_compare(best_val_gen_loss):
                    best_checkpoint_path = checkpoint_path
                    best_val_gen_loss = val_gen_loss
                    print('found best checkpoint', checkpoint['current_epoch'], checkpoint['current_iteration'])
                    print('val gen loss', validation_task, validation_key, get_loss_to_compare(val_gen_loss))
        else:
            assert validation_task is None and validation_key is None, (validation_task, validation_key)
            best_checkpoint_path = os.path.join(checkpoint_logdir, 'best.pt')
            if not os.path.exists(best_checkpoint_path):
                print('[INFO] cannot found validation best checkpoint')
                best_checkpoint_path = None

        checkpoint_paths = []
        if best_checkpoint_path is not None:
            checkpoint_paths.append(best_checkpoint_path)
        # get latest
        if os.path.exists(os.path.join(checkpoint_logdir, 'latest_checkpoint.txt')):
            fn = os.path.join(checkpoint_logdir, 'latest_checkpoint.txt')
            with open(fn, 'r') as f:
                line = f.read().splitlines()
            latest_checkpoint_path = os.path.join(checkpoint_logdir, line[0].split(' ')[-1])
            checkpoint_paths.append(latest_checkpoint_path)

    print('found checkpoints')
    print('\n'.join(checkpoint_paths))

    with open(os.path.join(checkpoint_logdir, 'cfg.json'), 'r') as f:
        cfg = AttrDict(json.load(f))

    update_legacy_config(cfg)
    net_G, *_ = get_model_optimizer_and_scheduler(cfg, seed=seed)

    if hasattr(net_G.module.mpi_embedding, 'theta_bias'):
        net_G.module.mpi_embedding.theta_bias = net_G.module.mpi_embedding.theta_bias.clone()  # hack, otherwise errors on cpu

    # print(BaseTrainer.load_checkpoint(AttrDict(net_G=net_G), cfg,
    #                                   checkpoint_path=os.path.join(checkpoint_logdir, 'epoch_00009_iteration_000016866_checkpoint.pt'),
    #                                   resume=False))
    #
    if hasattr(net_G.module, 'supervised_depth_loss'):
        net_G.module.supervised_depth_loss._init_model()
    if hasattr(net_G.module, 'flow_loss'):
        net_G.module.flow_loss._init_model()

    for t in net_G.parameters():
        t.requires_grad = False

    net_G.eval()

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print('loading checkpoint', checkpoint_path)
        epoch, iteration = BaseTrainer.load_checkpoint(AttrDict(net_G=net_G), cfg,
                                                       checkpoint_path=checkpoint_path,
                                                       resume=False, strict=strict)
        print(epoch, iteration)  # FIXME: not corresponding to the checkpoint, always the latest epoch, iteration?

        # for t in net_G.parameters():
        #     assert not t.requires_grad

        yield cfg, net_G, dict(epoch=epoch, iteration=iteration,
                               checkpoint_path=checkpoint_path,
                               next_checkpoint_path=checkpoint_paths[(i+1) % len(checkpoint_paths)],
                               prev_checkpoint_path=checkpoint_paths[(i-1) % len(checkpoint_paths)])

    del net_G
    import gc
    gc.collect()


class ViewSynthesisHelper(BaseHTMLHelper):
    # The reference camera position can just be the identity

    # # Accurate intrinsics are only important if we are trying to match a ground
    # # truth output. Here we just give intrinsics for a 16:9 image with the
    # # principal point in the center.
    # intrinsics = [1.0, 1.0 * 16/9, 0.5, 0.5]
    # FIXME: !!! change intrinsics based on image size ratio

    def path_planning(self, num_frames, x, y, z, path_type='circle', **unused):
        # https://github.com/vt-vl-lab/3d-photo-inpainting/blob/de0446740a3726f3de76c32e78b43bd985d987f9/utils.py#L29

        if path_type == 'straight_line':
            corner_points = np.array([[0, 0, 0], [(0 + x) * 0.5, (0 + y) * 0.5, (0 + z) * 0.5], [x, y, z]])
            corner_t = np.linspace(0, 1, len(corner_points))
            t = np.linspace(0, 1, num_frames)
            cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
            spline = cs(t)
            xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
        elif path_type == 'double_straight_line':
            corner_points = np.array([[-x, -y, -z], [0, 0, 0], [x, y, z]])
            corner_t = np.linspace(0, 1, len(corner_points))
            t = np.linspace(0, 1, num_frames)
            cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
            spline = cs(t)
            xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
        elif path_type == 'circle':
            xs, ys, zs = [], [], []
            for frame_id, bs_shift_val in enumerate(np.arange(-2.0, 2.0, (4. / num_frames))):
                xs += [np.cos(bs_shift_val * np.pi) * 1 * x]
                ys += [np.sin(bs_shift_val * np.pi) * 1 * y]
                zs += [np.cos(bs_shift_val * np.pi / 2.) * 1 * z]
            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        else:
            raise NotImplementedError(path_type)

        return xs, ys, zs

    def render_sway(self, layers, vis, depths, intrinsics, configs, table_name='view_syn',
                    apply_crop_fn=None, return_frames=False):
        # Render sway path
        layout = [[]]
        default_config = dict(
            num_frames=128, fps=8,
        )
        # configs = configs or [
        #     dict(path_type='circle', x=0.04, y=0.02, z=0.04, fps=8),
        #     dict(path_type='double_straight_line', x=0.1, y=0.0, z=0.0, fps=64),
        # ]
        for config in configs:
            config = default_config | config
            frames = []
            xs, ys, zs = self.path_planning(**config)
            for i in range(config['num_frames']):
                i_output = self.render(layers, xs[i], ys[i], zs[i], depths=depths, intrinsics=intrinsics)
                if apply_crop_fn:
                    i_output = apply_crop_fn(i_output)
                i_output = collect_tensor(i_output)
                frames.append(i_output)
            layout[0].append(dict(video=frames, fps=config['fps']))
        if return_frames:
            return [layout[0][i]['video'] for i in range(len(layout[0]))]

        self.dump_table(vis=vis, layout=layout, col_type="video", table_name=table_name)

    @staticmethod
    def render(layers: torch.Tensor, x_offset, y_offset, z_offset, depths, intrinsics):
        """

        Args:
            layers: [L, C+1, H, W], in range [0, 1]
            target_pose:
            depths: [L]

        Returns:

        """
        # reference_pose = torch.tensor(self.reference_pose, device=layers.device)
        # intrinsics = torch.tensor(self.intrinsics, device=layers.device)
        reference_pose = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]], device=layers.device, dtype=layers.dtype)

        target_pose = torch.tensor(
            [[1.0, 0.0, 0.0, -x_offset],
             [0.0, 1.0, 0.0, -y_offset],
             [0.0, 0.0, 1.0, -z_offset]], device=layers.device, dtype=layers.dtype)

        new_layers = mpi.render_layers(layers.movedim(-3, -1), depths,
                                       reference_pose, intrinsics,  # Reference view
                                       target_pose, intrinsics,  # Target view
                                       ).movedim(-1, -3)
        image = mpi.compose_back_to_front(new_layers)
        return image

    def view_static(self, layers, vis, depths=None):
        """

        Args:
            layers: [L, C+1, H, W]
            vis:
            depths: [L]

        Returns:

        """

        row = []
        for i in range(5):
            xoffset = (i - 2) * 0.05
            row.append(dict(image=Image.fromarray(self.render(layers, xoffset, 0.0, 0.0)),
                            info=f"xoff = {xoffset}"))

        for i in range(5):
            zoffset = (i - 2) * 0.15
            row.append(dict(image=Image.fromarray(self.render(layers, 0.0, 0.0, zoffset)),
                            info=f"zoff = {zoffset}"))

        self.dump_table(vis=vis, layout=[row], col_type='image', table_name='view_syn')

    def view_dynamic(self, layers, vis, depths=None):
        # if vis not in self.vis_seen:
        #     self.vis_seen.add(vis)
        # vis._print(self.mpi_style)
        if depths is None:
            depths = self.make_default_depths(layers)

        # only support no batch or batch size = 1
        if len(layers.shape) == 5:
            layers = layers.squeeze(0)

        html = []
        html.append(self.mpi_style)
        html.append('<div id=view><div id=mpi>')
        for i in range(len(depths)):
            depth = depths[i]
            url = imgurl(layers[i])
            html.append('''
            <div class=layer
                 style="transform: scale(%.3f) translateZ(-%.3fpx);
                 background-image: url(%s)"></div>''' % (depth, depth, url))

        html.append('</div></div>')
        html.append(self.MPI_JS)
        html = ''.join(html)
        # vis._print(html)
        self.dump_table(vis=vis, layout=[[html]], col_type='raw', table_name='view_syn')

    @staticmethod
    def make_default_depths(layers, front_depth=1, back_depth=100):
        depths = mpi.make_depths(front_depth, back_depth, layers.shape[-4]).to(layers.device)
        return depths

    @property
    def mpi_style(self):
        return '''
  <style>
  #view {
    position: relative;
    overflow: hidden;
    border: 2px solid black;
    height: %dpx;
    width: %dpx;
    perspective: 500px;
    background: #000;
  }''' % tuple(self.img_size) + '''
    #mpi {
      transform-style: preserve-3d; -webkit-transform-style: preserve-3d;
      height: 100%;
      width: 100%;
      pointer-events: none;
    }
    .layer {
      position: absolute;
      background-size: 100% 100%;
      background-repeat: no-repeat;
      background-position: center;
      width: 100%;
      height: 100%;
    }
    </style>
'''

    SPEED = 5
    MPI_JS = '''
  <script>
  function setView(mpi, x, y) {
    x = 2*x - 1;
    y = 2*y - 1;
    rx = (-1.5 * y).toFixed(2);
    ry = (2.0 * x).toFixed(2);
    // Put whatever CSS transform you want in here.
    mpi.style.transform =
        `rotateX(${rx}deg) rotateY(${ry}deg) translateZ(500px) scaleZ(500)`;
  }

  view = document.querySelector('#view');
  mpi = document.querySelector('#mpi');
  setView(mpi, 0.5, 0.5);

  // View animates by itself, or you can hover over the image to control it.
  let t = 0;
  let animate = true;
  function tick() {
    if (!animate) {
      return;
    }
    t = ''' + '(t + %d)' % SPEED + ''' % 300;
    r = Math.PI * 2 * t / 300;
    setView(mpi, 0.5 + 0.3 * Math.cos(r), 0.5 + 0.3 * Math.sin(r));
    requestAnimationFrame(tick);
  }
  tick();

  view.addEventListener('mousemove',
    (e) => {animate=false; setView(mpi, e.offsetX/view.offsetWidth, e.offsetY/view.offsetHeight);});
  view.addEventListener('mouseleave',
    (e) => {animate=true; tick();});
  </script>
  '''

def imgurl(image):
    # We resize layers to 512x288 so the whole stack can be serialized in a
    # Colab for the HTML viewer without hitting the memory restriction. Outside
    # Colab there is no such restriction and 512x512 layers could be used.
    #   image = tf.image.resize(image, [288, 512], method='area')
    #   data = tf.image.encode_png(
    #       tf.image.convert_image_dtype(image, tf.uint8)).numpy()
    #   data = denormalize_rgb(rgba_to_rgb(image))
    assert image.shape[-3] in [3, 4], image.shape
    image = collect_tensor(image, padding=0)

    buffered = io.BytesIO()
    Image.fromarray(image, 'RGBA').save(buffered, format="PNG")
    data = buffered.getvalue()
    dataurl = 'data:image/png;base64,{}'.format(base64.b64encode(data).decode())
    return dataurl
