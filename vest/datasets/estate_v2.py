import os
from tqdm import tqdm
import torch
import pickle
import numpy as np
from common import DATASET_DIR, ROOT
from vest.datasets.clevrer import MyBaseDataset
import imageio
import cv2


def get_downloaded_clips(root_dir, split):
    if split == 'val':
        return []  # FIXME
    successful_clips = []
    num_jobs = {'ibrnet_subset': 4, 'test': 8, 'train': 128}[split]
    for job_ind in range(num_jobs):
        suffix = f"_job_{job_ind}_{num_jobs}"
        if not os.path.exists(os.path.join(root_dir, split, "frames", f'successful_clips{suffix}.txt')):
            # raise RuntimeError(f"{root_dir}, {split}, {suffix}")
            print(f"[ERROR] {root_dir}, {split}, {suffix}")
            continue
        with open(os.path.join(root_dir, split, "frames", f'successful_clips{suffix}.txt'), 'r') as f:
            for clip in f.readlines():
                successful_clips.append(clip.strip('\n'))
    print('[INFO] number of clips successfully downloaded', len(successful_clips))

    return successful_clips


class Dataset(MyBaseDataset):
    root_dir = os.path.join(DATASET_DIR, 'estate')
    # print('loading from', root_dir)

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference, is_test)

        self.rgb_files, self.indices = self.get_rgb_files(cfg)

        print(f'found {len(self.rgb_files)} videos, {len(self.indices)} clips')

    def get_rgb_files(self, cfg):
        # TODO: use self.data_info.split
        # WARNINGS: assume that cfg is uniquely identified by cfg.name
        cache_file = os.path.join(
            DATASET_DIR, 'estate', f"metadata_{cfg.data.name}_{self.data_info.split}.pkl")
        if not os.path.exists(cache_file):
            print('creating cache', cache_file)

            all_rgb_files = []
            for scene_path in tqdm(get_downloaded_clips(self.data_info.split)):
            # for scene_path in os.listdir(os.path.join(self.root_dir, self.data_info.split)):
            #     if scene_path.startswith('failed') or scene_path.startswith('successful'):
            #         continue
                scene_path = os.path.join(self.root_dir, self.data_info.split, scene_path)
                rgb_files = [os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))]
                timestamps = [int(os.path.basename(rgb_file).split('.')[0]) for rgb_file in rgb_files]

                # check that the number of extracted frames matches camera param file from the raw dataset
                video_id = os.path.basename(scene_path)
                cam_file = os.path.join(DATASET_DIR, 'estate_raw', self.data_info.split, f"{video_id}.txt")
                with open(cam_file, 'r') as f:
                    num_frames = len(f.readlines()) - 1  # 1 line for youtube url
                    if num_frames != len(rgb_files):
                        # print('extracted', len(rgb_files), 'actual', num_frames, scene_path)
                        # raise RuntimeError()
                        # there are a lot
                        # FIXME: ignore the error for now
                        continue

                sorted_ids = np.argsort(timestamps)
                rgb_files = np.array(rgb_files)[sorted_ids]
                timestamps = np.array(timestamps)[sorted_ids]
                assert (timestamps == sorted(timestamps)).all()

                # all_rgb_files.append((rgb_files, timestamps))
                all_rgb_files.append([(rgb_files[i], timestamps[i]) for i in range(len(rgb_files))])

            with open(cache_file, 'wb') as f:
                pickle.dump(all_rgb_files, f)

            # write indices
            cache_file = os.path.join(
                DATASET_DIR, 'estate', f"indices_{cfg.data.name}_{self.data_info.split}.pkl")
            print('creating cache', cache_file)

            num_files = len(sum(all_rgb_files, []))
            num_distinct_files = len(set(sum(all_rgb_files, [])))
            assert num_files == num_distinct_files, (num_files, num_distinct_files)
            print(f'found {num_files} rgb files')

            # must regenerate indices

            indices = []
            for i, rgb_files_this_video in enumerate(all_rgb_files):
                if len(rgb_files_this_video) < self.sequence_length_max:
                    print(f'found sequence length {len(rgb_files_this_video)} but need {self.sequence_length_max}')
                    continue
                start_t_candidates = np.arange(len(rgb_files_this_video) - self.sequence_length_max + 1)
                # start_t_candidates = range(0, len(rgb_files_this_video) - self.sequence_length_max + 1, self.sequence_length_max)
                indices.extend([(i, j) for j in start_t_candidates])

            with open(cache_file, 'wb') as f:
                pickle.dump(indices, f)

        print('loading metadata from cache', cache_file)
        with open(cache_file, 'rb') as f:
            all_rgb_files = pickle.load(f)

        cache_file = os.path.join(
            DATASET_DIR, 'estate', f"indices_{cfg.data.name}_{self.data_info.split}.pkl")
        print('loading indices from cache', cache_file)
        with open(cache_file, 'rb') as f:
            indices = pickle.load(f)

        return all_rgb_files, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        video_ind, start_t = self.indices[index]
        start_t += np.random.choice(self.sequence_length_max - self.sequence_length + 1)

        # extract all camera parameters for this video
        # for f, _ in self.rgb_files[video_ind]:
        #     assert os.path.dirname(f) == os.path.dirname(self.rgb_files[video_ind][0][0])
        # cam_file = os.path.dirname(self.rgb_files[video_ind][0]).replace('frames', 'cameras') + '.txt'
        video_id = os.path.basename(os.path.dirname(self.rgb_files[video_ind][0][0]))
        cam_file = os.path.join(DATASET_DIR, 'estate_raw', self.data_info.split, f"{video_id}.txt")
        cam_params = {}

        # TODO: this is very inefficient
        # cache this in init
        with open(cam_file, 'r') as f:
            for line_ind, line in enumerate(f.readlines()):
                if line_ind == 0:
                    # first line is youtube url
                    continue
                entry = [float(x) for x in line.split()]
                timestamp = int(entry[0])
                intrinsics = np.array(entry[1:5], dtype=np.float32)  # (4,)
                # poses = np.array([entry[7:11], entry[11:15], entry[15:19]])  # (3, 4)  # world to camera
                poses = np.array(entry[7:], dtype=np.float32).reshape(3, 4)
                cam_params[timestamp] = (intrinsics, poses)

        def get_data_for_id(id):
            rgb_file, timestamp = self.rgb_files[video_ind][id]
            # print(rgb_file, timestamp)
            # should correspond to each olther

            # read rgbs
            rgb = imageio.imread(rgb_file)  # (480, 640, 3)
            if rgb.shape[0] != self.img_size[0] or rgb.shape[1] != self.img_size[1]:
                rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)
            rgb = rgb.astype(np.float32) / 255.

            K, pose = cam_params[timestamp]
            return rgb, K, pose

        # source image is start_t + self.sequence_length - 2 i.e. the last input frames
        id_src = start_t + self.sequence_length - 2

        num_frames = len(self.rgb_files[video_ind])
        id_qry = select_id_query_stereo_mag(num_frames, id_src)
        # print('src qry gap', id_src - id_qry)

        sequence_data = list(map(get_data_for_id, range(start_t, start_t + self.sequence_length)))
        query_data = get_data_for_id(id_qry)

        data = dict()
        data['images'] = torch.from_numpy(np.stack([d[0] for d in sequence_data])).movedim(-1, -3)
        data['cameras'] = torch.from_numpy(np.stack([d[1] for d in sequence_data]))
        data['poses'] = torch.from_numpy(np.stack([d[2] for d in sequence_data]))
        data['query_image'] = torch.from_numpy(query_data[0]).movedim(-1, -3)
        data['query_camera'] = torch.from_numpy(query_data[1])
        data['query_pose'] = torch.from_numpy(query_data[2])

        # TODO: tune front, back depth in config
        # # get depth range
        # depth_range = torch.tensor([1., 100.])

        if self.normalize:
            # [0, 1] -> ([-1, 1]
            data['images'] = data['images'] * 2 - 1
            data['query_image'] = data['query_image'] * 2 - 1
        return data


def select_id_query_IBRNet(num_frames, id_src):
    """

    Args:
        num_frames: number of timestamps in this clip
        id_src: selected timestamp as view synthesis source, implemented as I_t

    Returns:
        id_qry: selected timestamp as view synthesis target
    """
    # choose target view for view synthesis
    # in KITTI it's the stereo view for id_src
    window_size = 10
    # shift = np.random.randint(low=-1, high=2)
    # id_render = np.random.randint(low=4, high=num_frames-4-1)

    # right_bound = min(id_src + window_size + shift, num_frames-1)
    right_bound = min(id_src + window_size, num_frames - 1)
    left_bound = max(0, right_bound - 2 * window_size)
    # candidate_ids = np.arange(left_bound, right_bound)
    # remove the query frame itself with high probability
    # if np.random.choice([0, 1], p=[0.01, 0.99]):
    # candidate_ids = candidate_ids[candidate_ids != id_src]
    id_qry = np.random.choice(list(range(left_bound, id_src)) + list(range(id_src+1, right_bound)))

    return id_qry


def select_id_query_larger_window(num_frames, id_src):
    window_size = 30
    right_bound = min(id_src + window_size, num_frames - 1)
    left_bound = max(0, right_bound - 2 * window_size)
    id_qry = np.random.choice(list(range(left_bound, id_src)) + list(range(id_src+1, right_bound)))

    return id_qry


def select_id_query_stereo_mag(num_frames, id_src, sample_strategy):
    window_sign = -1 if np.random.random() < 0.5 else 1
    if sample_strategy == '5':
        id_qry = id_src + window_sign * 5
    if sample_strategy == '10':
        id_qry = id_src + window_sign * 10
    if sample_strategy == 'random':
        id_qry = id_src + window_sign * np.random.randint(low=1, high=30)
    if sample_strategy == 'random_positive':
        id_qry = id_src + np.random.randint(low=1, high=31)
    if sample_strategy == 'random_10':
        id_qry = id_src + window_sign * np.random.randint(low=1, high=10)
    if sample_strategy == 'random_5_positive':
        id_qry = id_src + np.random.randint(low=1, high=6)
    if sample_strategy == 'random_10_positive':
        id_qry = id_src + np.random.randint(low=1, high=11)
    if sample_strategy == '5_or_10':
        p = np.random.random()
        if p < 0.5:
            id_qry = id_src + window_sign * 10
        else:
            id_qry = id_src + window_sign * 5
    if sample_strategy == '5_or_10_or_random':
        p = np.random.random()
        if p < 0.33:
            id_qry = id_src + window_sign * 10
        elif p < 0.66:
            id_qry = id_src + window_sign * 5
        else:
            id_qry = id_src + window_sign * np.random.randint(low=1, high=30)

    id_qry = min(id_qry, num_frames - 1)
    id_qry = max(id_qry, 0)

    return id_qry


# https://github.com/googleinterns/IBRNet/blob/master/ibrnet/data_loaders/realestate.py


if __name__ == "__main__":
    from vest.config import get_attr_dict
    cfg = get_attr_dict(os.path.join(ROOT, 'configs/datasets/estate_with_cam.yaml'))
    print(cfg)

    d = Dataset(cfg)

    batch = d[0]
    for k, v in batch.items():
        print(k, v.shape)
    print()
