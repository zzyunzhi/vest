import fnmatch
import os
import numpy as np
from vest.datasets.kitti_dataset import Dataset as MonoDepth2Dataset
from vest.models.monodepth2 import STEREO_SCALE_FACTOR
from tu.ddp import master_only_print
import torch


def raw_city_sequences():
    """Sequence names for city sequences in kitti raw data.

    Returns:
      seq_names: list of names
    """
    seq_names = [
        '2011_09_26_drive_0001',
        '2011_09_26_drive_0002',
        '2011_09_26_drive_0005',
        '2011_09_26_drive_0009',
        '2011_09_26_drive_0011',
        '2011_09_26_drive_0013',
        '2011_09_26_drive_0014',
        '2011_09_26_drive_0017',
        '2011_09_26_drive_0018',
        '2011_09_26_drive_0048',
        '2011_09_26_drive_0051',
        '2011_09_26_drive_0056',
        '2011_09_26_drive_0057',
        '2011_09_26_drive_0059',
        '2011_09_26_drive_0060',
        '2011_09_26_drive_0084',
        '2011_09_26_drive_0091',
        '2011_09_26_drive_0093',
        '2011_09_26_drive_0095',
        '2011_09_26_drive_0096',
        '2011_09_26_drive_0104',
        '2011_09_26_drive_0106',
        '2011_09_26_drive_0113',
        '2011_09_26_drive_0117',
        '2011_09_28_drive_0001',
        '2011_09_28_drive_0002',
        '2011_09_29_drive_0026',
        '2011_09_29_drive_0071',
    ]
    return seq_names


def fvs_train_sequences():
    # https://github.com/YueWuHKUST/FutureVideoSynthesis/blob/main/doc/TestSetting.md
    seq_names = """
2011_09_26_drive_0001_sync  2011_09_26_drive_0018_sync  2011_09_26_drive_0104_sync
2011_09_26_drive_0002_sync  2011_09_26_drive_0048_sync  2011_09_26_drive_0106_sync
2011_09_26_drive_0005_sync  2011_09_26_drive_0051_sync  2011_09_26_drive_0113_sync
2011_09_26_drive_0009_sync  2011_09_26_drive_0056_sync  2011_09_26_drive_0117_sync
2011_09_26_drive_0011_sync  2011_09_26_drive_0057_sync  2011_09_28_drive_0001_sync
2011_09_26_drive_0013_sync  2011_09_26_drive_0059_sync  2011_09_28_drive_0002_sync
2011_09_26_drive_0014_sync  2011_09_26_drive_0091_sync  2011_09_29_drive_0026_sync
2011_09_26_drive_0017_sync  2011_09_26_drive_0095_sync  2011_09_29_drive_0071_sync
    """
    seq_names = seq_names.split()
    return seq_names


def fvs_test_sequences():
    seq_names = """
2011_09_26_drive_0060_sync  2011_09_26_drive_0084_sync  2011_09_26_drive_0093_sync  2011_09_26_drive_0096_sync
"""
    seq_names = seq_names.split()
    return seq_names


class LDIDataLoader(object):
    def preload_calib_files(self):
        """Preload calibration files for the sequence."""
        self.cam_calibration = {}
        assert self.dataset_variant == 'raw_city'
        seq_names = raw_city_sequences()
        for seq_id in seq_names:
            seq_date = seq_id[0:10]
            calib_file = os.path.join(self.root_dir, seq_date,
                                      'calib_cam_to_cam.txt')
            self.cam_calibration[seq_date] = self.read_calib_file(calib_file)

    def read_calib_file(self, file_path):
        """Read camera intrinsics."""
        # Taken from https://github.com/hunse/kitti
        float_chars = set('0123456789.e+- ')
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if key not in ['calib_time', 'corner_dist']:
                    data[key] = np.array(list(map(float, value.split(' '))))
                # if float_chars.issuperset(value):
                #     # try to cast to float array
                #     try:
                #         data[key] = np.array(map(float, value.split(' ')))
                #     except ValueError:
                #         # casting error: data[key] already eq. value, so pass
                #         pass

        return data


def get_raw_city_seq_dir(root_dir, data_split):
    exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.png'
    seq_names = raw_city_sequences()
    seq_indices = list(range(len(seq_names)))
    rng = np.random.RandomState(0)
    rng.shuffle(seq_names)
    rng.shuffle(seq_indices)  # FIXME: seq_indices won't correspond to seq_names because it's shuffled with next rng
    # FIXME: split will be different with elastic runs num_tries > 1
    print(seq_indices[:100])
    n_all = len(seq_names)
    n_train = int(round(0.7 * n_all))
    n_val = int(round(0.15 * n_all))
    seq_dirs = []
    if data_split == 'train':
        seq_names = seq_names[0:n_train]
    elif data_split == 'val':
        seq_names = seq_names[n_train:(n_train + n_val)]
    elif data_split == 'test':
        seq_names = seq_names[(n_train + n_val):n_all]
    for seq_id in seq_names:
        seq_date = seq_id[0:10]
        seq_dir = os.path.join(root_dir, seq_date,
                               '{}_sync'.format(seq_id))
        seq_dirs.append(seq_dir)
    return seq_dirs


class Dataset(MonoDepth2Dataset):
    # a wrapper for KITTIRAWDataset

    def __init__(self, cfg, is_inference=False, is_test=False):
        self.dataset_variant = cfg.data.generator_params.dataset_variant
        super().__init__(cfg, is_inference, is_test)
        root_dir = self.root_dir

        class DummyDataset(LDIDataLoader):
            def __init__(self):
                self.root_dir = root_dir
                self.dataset_variant = 'raw_city'
                self.h, self.w = cfg.data.img_size

        d = DummyDataset()
        # d.init_img_names_seq_list()
        d.preload_calib_files()
        self.ldi_dataset = d
        self.use_monodepth2_calibration = cfg.data.generator_params.use_monodepth2_calibration

        self.use_zero_tm1 = cfg.data.generator_params.use_zero_tm1
        master_only_print('[INFO] generator params', cfg.data.generator_params)

    def get_seq_dirs(self):
        master_only_print('[INFO] loading variant', self.dataset_variant, 'split', self.data_info.split)
        if self.dataset_variant == 'raw_city':
            return get_raw_city_seq_dir(self.root_dir, data_split=self.data_info.split)
        elif self.dataset_variant == 'fvs':
            if self.data_info.split == 'train':
                seq_names = fvs_train_sequences()
            elif self.data_info.split == 'val':
                seq_names = fvs_train_sequences()[:1]  # FIXME: dummy
            elif self.data_info.split == 'test':
                seq_names = fvs_test_sequences()
            seq_dirs =[]
            for seq_id in seq_names:
                seq_date = seq_id[0:10]
                seq_dir = os.path.join(self.root_dir, seq_date, seq_id)
                seq_dirs.append(seq_dir)
            return seq_dirs
        raise NotImplementedError(self.dataset_variant)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # full_res_h_w = (self.dataset.full_res_shape[1], self.dataset.full_res_shape[0])  # flipped
        full_res_h_w = data[(('color', 0, -1), 'full_res_h_w')][0].tolist()
        # should be the same for ('color', 's', -1) and time > 0
        del data[(('color', 0, -1), 'full_res_h_w')]
        del data[(('color', 's', -1), 'full_res_h_w')]
        if self.use_monodepth2_calibration:
            return data

        scene_path, _, side = self.scene_paths[index]
        seq_date = os.path.dirname(scene_path)  # e.g. scene_path = 2011_10_03/2011_01_03_drive_0027_sync

        new_data = {}
        new_data['images'] = data['images']
        new_data['stereo_images'] = data['stereo_images']

        if self.use_zero_tm1:
            new_data['images'][0] *= 0
            new_data['stereo_images'][0] *= 0

        calib_data = self.ldi_dataset.cam_calibration[seq_date]

        calib_src = calib_data['P_rect_02'].reshape(3, 4)
        fx, fy, px, py = calib_src[0, 0] / full_res_h_w[1], calib_src[1, 1] / full_res_h_w[0], \
                         calib_src[0, 2] / full_res_h_w[1], calib_src[1, 2] / full_res_h_w[0]
        camera = np.array([fx, fy, px, py])
        new_data[('K', 0)] = torch.Tensor(camera).view(1, 4).expand(self.sequence_length, 4)

        k_s = np.copy(calib_data['P_rect_02'].reshape(3, 4)[:3, :3])
        k_t = np.copy(calib_data['P_rect_03'].reshape(3, 4)[:3, :3])
        trans_src = np.copy(calib_data['P_rect_02'].reshape(3, 4)[:, 3])
        trans_trg = np.copy(calib_data['P_rect_03'].reshape(3, 4)[:, 3])

        # The translation is in homogeneous 2D coords, convert to regular 3d space:
        trans_src[0] = (trans_src[0] - k_s[0, 2] * trans_src[2]) / k_s[0, 0]
        trans_src[1] = (trans_src[1] - k_s[1, 2] * trans_src[2]) / k_s[1, 1]

        trans_trg[0] = (trans_trg[0] - k_t[0, 2] * trans_trg[2]) / k_t[0, 0]
        trans_trg[1] = (trans_trg[1] - k_t[1, 2] * trans_trg[2]) / k_t[1, 1]

        trans = trans_trg - trans_src
        trans = trans * (1 if side == 'l' else -1) * (1 if not data['do_flip'] else -1) / STEREO_SCALE_FACTOR  # instead of trans * side_sign * baseline_sign! trans is already negative
        pose = np.concatenate([np.eye(3), trans.reshape(3, 1)], axis=-1)
        assert pose.shape == (3, 4)

        new_data['stereo_T'] = torch.Tensor(pose).view(1, 3, 4).expand(self.sequence_length, 3, 4)

        stereo_T_sign = 1 if data['stereo_T'][0, 0, 3] > 0 else -1
        if stereo_T_sign != (1 if new_data['stereo_T'][0, 0, 3] > 0 else -1):
            print('     stereo sign not matched!!')
            print(scene_path, 'side', side, 'old sign', stereo_T_sign)
            print('loaded translation', trans)  # sign should be consistent as monodepth2, FIXME: assume there is no flip

            import ipdb; ipdb.set_trace()
            print()

        return new_data
