import os
import sys
import gc
import lmdb
# from scripts.convert_to_lmdb_from_estate_format_v2 import decode_colmap
import json
import time
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import torchvision
from PIL import Image
import torch
import numpy as np
from vest.datasets.clevrer import MyBaseDataset
# from vest.datasets.estate_v2 import select_id_query_stereo_mag
# from vest.datasets import colmap_utils
from tu.ddp import master_only_print


class Dataset(MyBaseDataset):
    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference, is_test)
        self.dataset_name = cfg.data.dataset_name
        try:
            self.resize_transform = torchvision.transforms.Resize(self.img_size, torchvision.transforms.InterpolationMode.BICUBIC)
        except AttributeError:
            # old torch version to be compatible with tensorflow==1.15
            # not loading images anyways
            self.resize_transform = None
        self.input_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                )
            ]
        )
        self.sample_strategy = self.data_info.sample_strategy
        self.counter = 0

        if self.data_info.use_lmdb:
            self.lmdb_paths: List[str] = self.data_info.lmdb_paths
            self.rgb_lmdb_paths: List[str] = self.data_info.rgb_lmdb_paths
            self.envs = self._open_envs()
            self.rgb_envs = self._open_rgb_envs()

            if self.data_info.cache_file != '':
                cache_file = self.data_info.cache_file
                master_only_print('[INFO] loading metadata from cache', cache_file)
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                self.clip_to_image_ids_sorted_by_timestamps = cache['clip_to_image_ids_sorted_by_timestamps']
                # self.files = cache['files']

                # print('[INFO] number of scenes found under colmap', len(self.files))
                # self.files = self.files[:max(int(self.data_info.load_percentage * len(self.files)), 1)]
                # print('[INFO] take load_percentage', len(self.files))
                self.files = list(sorted(self.clip_to_image_ids_sorted_by_timestamps.keys()))
                print('[INFO] ignore load_percentage, number of scenes found under colmap', len(self.files))
                del cache
                gc.collect()

                if self.is_inference and not self.is_test:
                    # for fast validation, do this after considering load_percentage
                    self.files = self.files[:1000]
                return

            self.clip_to_image_ids_sorted_by_timestamps = dict()
            self.files: List[Tuple[str, Union[int, None], Union[int, None]]] = []
            stats = {'mean_num_points_by_clip': [],
                     'min_num_points_by_clip': [],
                     'max_num_points_by_clip': [],
                     'sequence_length': []}
            count_succ, count_fail = 0, 0
            for env in self.envs:
                with env.begin(write=False) as txn:
                    # keys, values = list(txn.cursor().iternext())
                    pbar = tqdm(total=txn.stat()['entries'])
                    for key, value in txn.cursor():
                        pbar.update(1)
                        _, colmap_images, _ = decode_colmap(value)

                        image_id_to_timestamps = dict()
                        success = True
                        num_points_this_clip = []
                        for image_id, image in colmap_images.items():
                            if sys.version_info.major == 3 and sys.version_info.minor <= 9:
                                # old python
                                assert image.name.endswith('.png')
                                image_id_to_timestamps[image_id] = int(image.name[:-len('.png')])
                            else:
                                image_id_to_timestamps[image_id] = int(image.name.removesuffix('.png'))
                            num_points = len([p for p in image.point3D_ids if p != -1])
                            num_points_this_clip.append(num_points)
                            if num_points < 250:  # FIXME: ?
                                # print(f"[ERROR] colmap < 250 points", num_points, len(image.point3D_ids))
                                success = False
                                break
                        if not success:
                            count_fail += 1
                            continue

                        stats['mean_num_points_by_clip'].append(np.mean(num_points_this_clip))
                        stats['min_num_points_by_clip'].append(np.min(num_points_this_clip))
                        stats['max_num_points_by_clip'].append(np.max(num_points_this_clip))
                        stats['sequence_length'].append(len(colmap_images))
                        count_succ += 1

                        image_ids_sorted_by_timestamps = [image_id for image_id, _ in
                                                          sorted(image_id_to_timestamps.items(), key=lambda item: item[1])]  # is it correct
                        clip = key.decode('ascii')
                        self.clip_to_image_ids_sorted_by_timestamps[clip] = image_ids_sorted_by_timestamps
                        self.files += [(clip, src, None) for src in range(self.sequence_length_max - 2, len(colmap_images) - 1)]

                        # if os.getenv('DEBUG') == '1' and count_succ >= 10 and not self.is_inference:
                        #     print('[DEBUG] break early')
                        #     break

                pbar.close()
                print('[INFO] one env done: succ so far', count_succ, 'fail so far', count_fail)
                print('[INFO] colmap stats for this env')
                print(json.dumps({k: np.mean(v) for k, v in stats.items()}, sort_keys=True, indent=4))

            print('[INFO] total succ', count_succ, 'total fail', count_fail)
            cache = {
                'clip_to_image_ids_sorted_by_timestamps': self.clip_to_image_ids_sorted_by_timestamps,
                'files': self.files,
            }
            cache_file = os.path.join(
                f"/viscam/data/{self.dataset_name}/metadata_{cfg.data.name}_{self.data_info.split}_{time.time()}.pkl")
            print(f"[INFO] saving cache to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            del cache
            gc.collect()

            # if os.getenv('DEBUG') == '1':
            #     for index in tqdm(range(10)):
            #         self[index]

            print('[INFO] number of scenes found under colmap', len(self.files))
            self.files = self.files[:max(int(self.data_info.load_percentage * len(self.files)), 1)]
            print('[INFO] take load_percentage', len(self.files))

            self.files = list(sorted(self.clip_to_image_ids_sorted_by_timestamps.keys()))
            print('[INFO] ignore load_percentage, number of scenes found under colmap', len(self.files))
            if self.is_inference and not self.is_test:
                # for fast validation
                self.files = self.files[:1000]
            return

        self.root_dir = f'/scr-ssd/yzzhang/{self.dataset_name}'
        if not os.path.exists(os.path.join(self.root_dir, self.data_info.split, 'colmap')):
            if not self.is_test:
                assert os.getenv('DEBUG') == '1'
            print('[DEBUG] cannot find colmap files in local storage')
            self.root_dir = f'/viscam/data/{self.dataset_name}'

        if self.data_info.cache_file != '':
            cache_file = self.data_info.cache_file
            master_only_print('loading metadata from cache', cache_file)
            if cache_file.endswith('.pkl'):
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                self.clip_to_image_ids_sorted_by_timestamps = cache['clip_to_image_ids_sorted_by_timestamps']
                self.clip_to_colmap_cameras = cache['clip_to_colmap_cameras']
                self.clip_to_colmap_images = cache['clip_to_colmap_images']
                self.clip_to_colmap_points3D = cache['clip_to_colmap_points3D']
                self.files = cache['files']
            return

        self.clip_to_image_ids_sorted_by_timestamps = dict()
        self.clip_to_colmap_cameras = dict()
        self.clip_to_colmap_images = dict()
        self.clip_to_colmap_points3D = dict()

        if self.data_info.clip_ids_dir != '':
            assert self.data_info.clip_ids_txt == ''
            all_clips = os.listdir(self.data_info.clip_ids_dir)
        elif self.data_info.clip_ids_txt != '':
            all_clips = np.loadtxt(self.data_info.clip_ids_txt, dtype=str)
        # all_clips = os.listdir(os.path.join(self.root_dir, self.data_info.split, "colmap"))
        print('[INFO] number of scenes found under colmap', len(all_clips))
        all_clips = list(sorted(all_clips))[:max(int(self.data_info.load_percentage * len(all_clips)), 1)]
        print('[INFO] take load_percentage', len(all_clips))
        if os.getenv('DEBUG') == '1':
            print('[DEBUG] take 10 scenes')
            all_clips = all_clips[:10]
        stats = {'mean_num_points_by_clip': [],
                 'min_num_points_by_clip': [],
                 'max_num_points_by_clip': [],
                 'mean_sequence_length': []}
        for clip in tqdm(all_clips):
            colmap_db = os.path.join(self.root_dir, self.data_info.split, "colmap", clip, "sparse")
            try:
                colmap_cameras, colmap_images, colmap_points3D = colmap_utils.read_model(colmap_db, ext=".bin")
            except Exception as e:
                print('[ERROR] cannot read from colmap output', clip, e)
                continue
            assert len(colmap_cameras) == 1
            image_id_to_timestamps = dict()
            success = True
            num_points_this_clip = []
            for image_id, image in colmap_images.items():
                image_id_to_timestamps[image_id] = int(image.name.removesuffix('.png'))
                num_points = len([p for p in image.point3D_ids if p != -1])
                num_points_this_clip.append(num_points)
                if num_points < 250:  # FIXME: ?
                    print(f"[ERROR] colmap < 250 points", num_points, len(image.point3D_ids))
                    success = False
                    break
            if not success:
                # skip the scene if any frame has <= 250 points
                continue
            stats['mean_num_points_by_clip'].append(np.mean(num_points_this_clip))
            stats['min_num_points_by_clip'].append(np.min(num_points_this_clip))
            stats['max_num_points_by_clip'].append(np.max(num_points_this_clip))
            stats['mean_sequence_length'].append(len(colmap_images))

            image_ids_sorted_by_timestamps = [image_id for image_id, _ in
                                              sorted(image_id_to_timestamps.items(), key=lambda item: item[1])]  # is it correct

            self.clip_to_image_ids_sorted_by_timestamps[clip] = image_ids_sorted_by_timestamps
            self.clip_to_colmap_cameras[clip] = colmap_cameras
            self.clip_to_colmap_images[clip] = colmap_images
            self.clip_to_colmap_points3D[clip] = colmap_points3D
        new_all_clips = self.clip_to_colmap_cameras.keys()
        print('[INFO] number of clips successfully loaded', len(new_all_clips), '/', len(all_clips))
        all_clips = new_all_clips
        print('[INFO] colmap stats')
        print(json.dumps({k: np.mean(v) for k, v in stats.items()}, sort_keys=True, indent=4))

        if self.data_info.files_pkl != '':
            # acid baseline50
            assert dataset_name == 'acid' and self.is_inference
            with open(self.data_info.files_pkl, 'rb') as f:
                files = pickle.load(f)
            self.files: List[Tuple[str, Dict, Dict]] = files
            return

        files: List[Tuple[str, Union[int, None], Union[int, None]]] = []
        for clip in tqdm(all_clips):
            files.extend([(clip, src, None) for src in range(self.sequence_length_max - 2, len(self.clip_to_image_ids_sorted_by_timestamps[clip]) - 1)])

        self.files = files

        # if os.getenv('DEBUG') == '1':
        #     print('[DEBUG] try loading frist 10 samples')
        #     for index in range(10):
        #         self[index]

        if self.dataset_name != 'estate':
            return
        cache = {
            'clip_to_image_ids_sorted_by_timestamps': self.clip_to_image_ids_sorted_by_timestamps,
            'clip_to_colmap_cameras': self.clip_to_colmap_cameras,
            'clip_to_colmap_images': self.clip_to_colmap_images,
            'clip_to_colmap_points3D': self.clip_to_colmap_points3D,
            'files': files,
        }
        cache_file = os.path.join(
            self.root_dir, f"metadata_{cfg.data.name}_{self.data_info.split}_{time.time()}.pkl")
        print(f"[INFO] saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def _open_envs(self):
        envs = []
        for fn in self.lmdb_paths:
            env = lmdb.open(fn, readonly=True, lock=False, max_readers=126,
                            readahead=False, meminit=False)
            envs.append(env)
        return envs

    def _open_rgb_envs(self):
        envs = []
        for fn in self.rgb_lmdb_paths:
            env = lmdb.open(fn, readonly=True, lock=False, max_readers=126,
                            readahead=False, meminit=False)
            envs.append(env)
        return envs

    def __len__(self):
        return len(self.files)

    def get_rgb_for(self, clip: str, timestamp: int):
        if self.data_info.use_lmdb:
            image_path = f'/scr-ssd/yzzhang/{self.dataset_name}/{self.data_info.split}/frames/{clip}/{timestamp}.png'
            if self.data_info.legacy_image_path:
                image_path = f'/scr-ssd/yzzhang/{self.dataset_name}/{self.data_info.split}_w{self.img_size[1]}_h{self.img_size[0]}/frames/{clip}/{timestamp}.png'
            for env in self.rgb_envs:  # very inefficient, but usually there are only two rgb envs so it might be fine
                with env.begin(write=False) as txn:
                    rgb = txn.get(image_path.encode('ascii'))
                if rgb is not None:
                    rgb = Image.frombytes(
                        'RGB',
                        size=(self.data_info.rgb_cache_file_width, self.data_info.rgb_cache_file_height),
                        data=rgb)
                    rgb = self.input_transform(rgb)
                    return image_path, rgb
            import ipdb; ipdb.set_trace()

        # image path without downasampling
        rgb_file = os.path.join(self.root_dir, self.data_info.split, "frames", clip, f"{timestamp}.png")
        if not os.path.exists(rgb_file):
            rgb_file = os.path.join('/viscam/data', os.path.relpath(rgb_file, '/scr-ssd/yzzhang'))
        image_path = rgb_file

        rgb_file = os.path.join(self.root_dir, f"{self.data_info.split}_w{self.img_size[1]}_h{self.img_size[0]}", "frames",
                                clip, f"{timestamp}.png")
        if not os.path.exists(rgb_file):
            local_rgb_file = os.path.join(self.root_dir, self.data_info.split, "frames", clip, f"{timestamp}.png")
            rgb_file = os.path.join('/viscam/data', os.path.relpath(local_rgb_file, '/scr-ssd/yzzhang'))
            if self.data_info.split == 'ibrnet_subset':
                if os.path.exists(local_rgb_file):
                    rgb_file = local_rgb_file
                else:
                    print('[ERROR] reading from non-local', rgb_file)
            else:
                if not self.is_test:
                    print('[ERROR] reading from non-local', rgb_file)
                    import ipdb; ipdb.set_trace()
                    raise RuntimeError()
            rgb = Image.open(rgb_file)
            rgb = self.resize_transform(rgb)
        else:
            rgb = Image.open(rgb_file)
        rgb = self.input_transform(rgb)
        return image_path, rgb

    def get_data_for(self, clip, image_id, colmap_images, colmap_cameras, colmap_points3D):
        data = dict()
        img_item = colmap_images[image_id]
        if sys.version_info.major == 3 and sys.version_info.minor <= 9:
            # old python
            # hack: not loading the 360p image because it won't be found on this machine
            assert img_item.name.endswith('.png')
            timestamp = int(img_item.name[:-len('.png')])
            if self.data_info.use_lmdb:
                image_path = f'/scr-ssd/yzzhang/{self.dataset_name}/{self.data_info.split}/frames/{clip}/{timestamp}.png'
                if self.data_info.legacy_image_path:
                    image_path = f'/scr-ssd/yzzhang/{self.dataset_name}/{self.data_info.split}_w{self.img_size[1]}_h{self.img_size[0]}/frames/{clip}/{timestamp}.png'
            data['timestamp'] = timestamp
            data['image_path'], data['image'] = image_path, torch.zeros(())
        else:
            timestamp = int(img_item.name.removesuffix('.png'))
            data['timestamp'] = timestamp
            data['image_path'], data['image'] = self.get_rgb_for(clip, timestamp)

        cam = colmap_cameras[img_item.camera_id]
        points3D = colmap_points3D
        # load sparse point cloud
        qvec = img_item.qvec
        tvec = img_item.tvec
        xyzs = [(point3D_id, xy, points3D[point3D_id].xyz)
                for xy, point3D_id in zip(img_item.xys, img_item.point3D_ids)
                if point3D_id != -1]
        colmap_camera = np.array([cam.params[0] / cam.width, cam.params[1] / cam.height,
                                  cam.params[2] / cam.width, cam.params[3] / cam.height], dtype=np.float32)  # (4,)
        R_cam_world = colmap_utils.qvec2rotmat(qvec).astype(np.float32)
        T_cam_world = np.array(tvec, dtype=np.float32).reshape(3, 1)
        colmap_pose = np.hstack((R_cam_world, T_cam_world))  # (3, 4)

        # use camera parameters from colmap model
        data['camera'] = colmap_camera
        data['pose'] = colmap_pose

        # Convert xyzs_world to homogeneous coordinates
        _, xys_cam, xyzs_world = zip(*xyzs)

        downsample_ratio_x = cam.width / self.img_size[1]
        downsample_ratio_y = cam.height / self.img_size[0]

        xys_cam = np.array(xys_cam).T.astype(np.float32)
        xys_cam[0] /= downsample_ratio_x
        xys_cam[1] /= downsample_ratio_y
        data['xys_cam'] = xys_cam

        xyzs_world = np.array(xyzs_world)
        xyzs_world_homo = np.hstack((xyzs_world,
                                     np.ones((len(xyzs_world), 1)))).T.astype(np.float32)
        # Transform xyzs to camera coordiantes
        G_cam_world = np.vstack([data['pose'], np.array([0, 0, 0, 1])]).astype(np.float32)
        xyzs_cam_homo = G_cam_world @ xyzs_world_homo
        xyzs_cam_homo /= xyzs_cam_homo[-1]
        data["xyzs"] = xyzs_cam_homo[:-1]

        if False:
            info = {'camera_params': cam.params}
            _info = {}
            _info['G_cam_world'] = G_cam_world
            f_x = info["camera_params"][0] / downsample_ratio_x
            f_y = info["camera_params"][1] / downsample_ratio_y
            p_x = info["camera_params"][2] / downsample_ratio_x
            p_y = info["camera_params"][3] / downsample_ratio_y
            _info["K"] = np.array([
                [f_x, 0, p_x],
                [0, f_y, p_y],
                [0, 0, 1]
            ], dtype=np.float32)
            I_Zero = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=np.float32)
            P = _info["K"] @ I_Zero @ _info["G_cam_world"]
            m_det_sign = np.sign(np.linalg.det(P[:, :-1]))
            m3_norm = np.linalg.norm(P[2][:-1])
            xys_cam_reproj = _info["K"] @ I_Zero @ xyzs_cam_homo
            depths = (m_det_sign * xys_cam_reproj[-1]) / m3_norm
            assert (depths == data['xyzs'][-1]).all()

        mask = data['xyzs'][2, :] >= 1e-3  # depth must be positive
        data['xys_cam'] = data['xys_cam'][:, mask]
        data['xyzs'] = data['xyzs'][:, mask]
        return data

    def __getitem__(self, index):
        self.counter += 1
        if self.counter % 1000 == 0:
            for env in self.envs + self.rgb_envs:
                env.close()
                del env
            self.envs = self._open_envs()
            self.rgb_envs = self._open_rgb_envs()
            torch.cuda.empty_cache()
            gc.collect()

        clip = self.files[index]
        line = None
        if isinstance(clip, tuple):
            clip, src, qry = clip
        else:
            assert isinstance(clip, str)
            if ' ' in clip:
                # clip (str) is a line in synsin test split .txt
                line = clip.split(' ')
                clip = str(line[0])
                # camera = np.array(map(float, line[3:7]))
                # pose_src = np.array(map(float, line[7:19])).reshape(3, 4)
                # pose_qry = np.array(map(float, line[19:31])).reshape(3, 4)
            else:
                clip, src, qry = clip, None, None

        image_ids = self.clip_to_image_ids_sorted_by_timestamps[clip]

        if self.data_info.use_lmdb:
            for env in self.envs:
                with env.begin(write=False) as txn:
                    value = txn.get(clip.encode('ascii'))
                    if value is not None:
                        break
            assert value is not None, (clip, src, qry, image_ids)
            colmap_cameras, colmap_images, colmap_points3D = decode_colmap(value)
            del value
        else:
            colmap_images = self.clip_to_colmap_images[clip]
            colmap_cameras = self.clip_to_colmap_cameras[clip]
            colmap_points3D = self.clip_to_colmap_points3D[clip]

        def get_data_for(image_id):
            return self.get_data_for(
                clip=clip, image_id=image_id,
                colmap_cameras=colmap_cameras, colmap_images=colmap_images, colmap_points3D=colmap_points3D)

        # if isinstance(src, dict):
        #     raise NotImplementedError()
        #     if 'image_id' in qry:
        #         # use colmap camera parameters when available
        #         query_data = get_data_for(image_id=qry['image_id'])
        #     else:
        #         query_data = qry
        #         query_data['image_path'], query_data['image'] = self.get_rgb_for(clip=clip, timestamp=qry['timestamp'])
        #
        #     if 'image_id' in src:
        #         src = image_ids.index(src['image_id'])
        #     else:
        #         src_data = self.get_rgb_for(clip, src['timestamp'])

        if line is not None:
            src, qry = None, None
            timestamp_src, timestamp_qry = int(line[1]), int(line[2])
            for image_id, image in colmap_images.items():
                if sys.version_info.major == 3 and sys.version_info.minor <= 9:
                    assert image.name.endswith('.png')
                    timestamp_this = image.name[:-len('.png')]
                else:
                    timestamp_this = image.name.removesuffix('.png')
                if int(timestamp_this) == timestamp_src:
                    src = image_ids.index(image_id)
                if int(timestamp_this) == timestamp_qry:
                    qry = image_ids.index(image_id)
                if src is not None and qry is not None:
                    break
            if src is None and qry is None:
                raise RuntimeError('[ERROR] cannot find timestamp in the clip')
            if not (0 <= src + 2 - self.sequence_length and src + 1 < len(image_ids)):
                print(f'[ERROR] src or qry out of bound: src {src}, qry {qry}, clip length {len(image_ids)}')
                src = max(src, self.sequence_length - 2)
                src = min(src, len(image_ids) - 2)
            # FIXME
            # if qry == src:
            #     print('qry == src, set qry = src + 1')
            #     qry = src + 1
            # qry = src + np.random.randint(low=-30, high=30)
            # qry = max(qry, 0)
            # qry = min(qry, len(image_ids) - 1)
            # qry = min(src + 30, qry)
            # qry = max(src - 30, qry)
            # if abs(qry - src) > 30:  # FIXME: happens a lot
            #     import ipdb; ipdb.set_trace()
            #     print('gap', qry - src, 'src', src, 'qry', qry, 'clip length', len(image_ids))
            # if not hasattr(self, 'stats'):
            #     self.stats = {'num_points_mean': [],
            #                   'num_points_min': [],
            #                   'num_points_max': []}
            # self.stats.append(abs(qry - src))
            # num_points_this_clip = []
            # for image in colmap_images.values():
            #     num_points = len([p for p in image.point3D_ids if p != -1])
            #     num_points_this_clip.append(num_points)
            # self.stats['num_points_mean'].append(np.mean(num_points_this_clip))
            # self.stats['num_points_min'].append(np.min(num_points_this_clip))
            # self.stats['num_points_max'].append(np.max(num_points_this_clip))
            # if len(self.stats['num_points_mean']) % 10 == 0:
            #     print('average number of points', {k: np.mean(v) for k, v in self.stats.items()})
        else:
            if src is None:
                src = np.random.randint(low=self.sequence_length - 2, high=len(colmap_images) - 1)
            if qry is None:
                qry = select_id_query_stereo_mag(len(image_ids), src,
                                                 sample_strategy=self.sample_strategy)
        query_data = get_data_for(image_id=image_ids[qry])

        # collect timestamps
        # sequence_data = list(map(lambda ind: self.get_data_for(clip, ind), range(src + 2 - self.sequence_length, src + 2)))
        # query_data = self.get_data_for(clip, qry)  # assume timestamps sorted

        # if not (src + 2 - self.sequence_length >= 0 and src + 2 <= len(image_ids)):
        #     #FIXME: deubg
        #     # import ipdb; ipdb.set_trace()
        #     src = np.random.randint(low=self.sequence_length - 2, high=len(colmap_images) - 1)

        assert src + 2 - self.sequence_length >= 0 and src + 2 <= len(image_ids), (src, self.sequence_length, len(image_ids))
        image_ids = [image_ids[ind] for ind in range(src + 2 - self.sequence_length, src + 2)]
        sequence_data = [get_data_for(image_id=image_id) for image_id in image_ids]
        if os.getenv('DEBUG') == '1':
            timestamps = [d['timestamp'] for d in sequence_data]
            if list(sorted(timestamps)) != timestamps:
                print("[ERROR]", timestamps, src)
                import ipdb; ipdb.set_trace()

        data = dict()
        # data['image_path'] = [d['image_path'] for d in sequence_data]
        data['images'] = torch.stack([d['image'] for d in sequence_data])
        data['cameras'] = torch.tensor(np.stack([d['camera'] for d in sequence_data]))
        data['poses'] = torch.tensor(np.stack([d['pose'] for d in sequence_data]))
        # if 'xyzs' in d:
        data['xyzs']: List = [torch.tensor(d['xyzs']) for d in sequence_data]
        data['xys_cam']: List = [torch.tensor(d['xys_cam']) for d in sequence_data]

        data['query_image'] = query_data['image']
        data['query_camera'] = torch.tensor(query_data['camera'])
        data['query_pose'] = torch.tensor(query_data['pose'])
        # data['query_xyzs'] = torch.from_numpy(query_data['xyzs'])
        # data['query_xys_cam'] = torch.from_numpy(query_data['xys_cam'])

        data['image_paths'] = [d['image_path'] for d in sequence_data]
        data['query_image_path'] = query_data['image_path']
        del sequence_data
        del query_data
        del colmap_images
        del colmap_cameras
        del colmap_points3D
        gc.collect()
        return data
