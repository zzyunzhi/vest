import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch.nn.functional as F
import torch
import glob
from vest.datasets.kitti_utils import generate_depth_map
from torchvision import transforms
from torchvision.transforms import functional as TF
from tu.ddp import master_only_print
from vest.datasets.mono_dataset import MonoDataset
from vest.datasets.clevrer import MyBaseDataset


KITTI_ROOT = '/svl/u/yzzhang/datasets/kitti'


class Dataset(MyBaseDataset):
    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference, is_test)
        self.root_dir = KITTI_ROOT

        master_only_print(cfg.data.generator_params)

        # get filenames relative to data_path

        self.use_stereo = cfg.data.generator_params.use_stereo
        assert self.use_stereo
        scene_paths = []
        for scene_path in self.get_seq_dirs():
            try:
                num_frames_left = len(os.listdir(os.path.join(scene_path, 'image_02', 'data')))
                num_frames_right = len(os.listdir(os.path.join(scene_path, 'image_03', 'data')))
            except FileNotFoundError as e:
                print(e)
                continue

            if num_frames_left != num_frames_right:
                print('number of frames not matched', num_frames_left, num_frames_right)
                continue

            if num_frames_left < self.sequence_length_max:
                print(f'found sequence length {num_frames_left} but need {self.sequence_length_max}')
                continue

            start_t_candidates = np.arange(num_frames_left - self.sequence_length_max + 1)
            # start_t_candidates = range(0, len(rgb_files_this_video) - self.sequence_length_max + 1, self.sequence_length_max)
            scene_path = os.path.relpath(scene_path, start=self.root_dir)
            scene_paths.extend([(scene_path, i, 'l') for i in start_t_candidates])
            scene_paths.extend([(scene_path, i, 'r') for i in start_t_candidates])
        self.scene_paths = scene_paths
        master_only_print('[INFO] found valid clips', len(scene_paths))

        self.dataset = KITTIRAWDataset(
            data_path=self.root_dir,
            filenames=[f"{scene_paths[0][0]} {scene_paths[0][1]:02d} {scene_paths[0][2]}"],  # dummy filename for check_depth
            height=self.img_size[0], width=self.img_size[1],
            frame_idxs=[0, 's'] if self.use_stereo else [0],
            num_scales=1,
            is_train=False,
            # WARNINGS: DO NOT set is_train to True with current implementation
            #  because do_flip is independent for timesteps in the same sequence
            img_ext='.png',
            interpolation=getattr(transforms.InterpolationMode, cfg.data.generator_params.interpolation),
        )
        # assert self.dataset.load_depth
        self.dataset.load_depth = False  # some ground truth points are missing, need to filter files (check_depth for each file)

        if not self.is_inference and not self.is_test:
            self.do_flip_p = cfg.data.generator_params.do_flip_p
            self.do_color_aug_p = cfg.data.generator_params.do_color_aug_p

    def get_seq_dirs(self):
        # ignore data_split, get all sequences available
        print('loading full kitti raw')
        return glob.glob(os.path.join(self.root_dir, '*', '*_sync'))

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, index):
        scene_path, frame_idx_start, side = self.scene_paths[index]
        frame_idx_start += np.random.choice(self.sequence_length_max - self.sequence_length + 1)

        self.dataset.filenames = [f"{scene_path} {frame_idx:02d} {side}"
                                  for frame_idx in range(frame_idx_start, frame_idx_start + self.sequence_length)]

        do_flip = not self.is_inference and not self.is_test and np.random.random() < self.do_flip_p
        do_color_aug = not self.is_inference and not self.is_test and np.random.random() < self.do_color_aug_p
        if do_color_aug:
            color_aug_params = transforms.ColorJitter.get_params(
                self.dataset.brightness, self.dataset.contrast, self.dataset.saturation, self.dataset.hue)

            def color_aug(img):
                # https://github.com/datumbox/vision/blob/12fd3a625a044a454cca3dbb2187e78efe1b4596/torchvision/transforms/transforms.py#L1167
                # forward without sampling
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = color_aug_params
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = TF.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = TF.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = TF.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = TF.adjust_hue(img, hue_factor)

                return img
        else:
            color_aug = (lambda x: x)
        # data_all_frames = list(self.dataset)
        # keep do_flip and do_color_aug consistent over all frames
        data_all_frames = [self.dataset._getitem(i, do_flip=do_flip, color_aug=color_aug) for i in range(len(self.dataset))]
        # list of dicts to dict of lists
        data = dict()
        for k in data_all_frames[0].keys():
            data[k] = torch.stack([data_this_frame[k] for data_this_frame in data_all_frames], dim=0)
        data['do_flip'] = do_flip

        for k in data.keys():
            if isinstance(k, tuple) and k[0] in ['color', 'color_aug']:
                if self.normalize:
                    data[k] = data[k] * 2 - 1  # [0, 1] -> [-1, 1]
        if 'depth_gt' in data:  # (t, 1, h, w)
            data['depth_gt'] = F.interpolate(
                data['depth_gt'], self.img_size, mode="bilinear", align_corners=False
            )

        # # data has keys ('color_aug', 0, 0), ('color_aug', 0, 's')
        # data['images'] = data[('color_aug', 0, 0)]
        # if self.use_stereo:
        #     data['stereo_images'] = data[('color_aug', 's', 0)]
        data['images'] = data.pop(('color_aug', 0, 0))
        data['stereo_images'] = data.pop(('color_aug', 's', 0))
        del data[('color', 0, 0)]
        del data[('color', 's', 0)]
        return data


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def preprocess(self, inputs, color_aug):
        # add a check to make sure intrinsics is using the correctfull_res_shape
        for k in list(inputs):
            if "color" in k:
                # assert inputs[k].size == self.full_res_shape, (k, inputs[k].size)  # not necessarily true
                inputs[(k, 'full_res_h_w')] = torch.tensor([inputs[k].size[1], inputs[k].size[0]])
        return super().preprocess(inputs, color_aug)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
