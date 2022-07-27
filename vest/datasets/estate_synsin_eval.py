import os
import glob
import time
import pickle
from tqdm import tqdm
from functools import partial
from tu.ddp import master_only_print
from typing import List, Dict, Tuple, Union
import torchvision
from PIL import Image
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict
try:
    from common import DATASET_DIR, ROOT
    from vest.datasets.clevrer import MyBaseDataset
except:
    # FIXME: remove this
    if __name__ == "__main__":
        import sys
        sys.path.insert(0, '/viscam/u/yzzhang/projects/img')
        from common import DATASET_DIR, ROOT
        from vest.datasets.clevrer import MyBaseDataset
# from vest.datasets.nerf_dataset import NeRFDataset
from vest.datasets.estate_v2 import select_id_query_stereo_mag, get_downloaded_clips
from vest.datasets import colmap_utils


def get_cam_params_for_clip(cam_file):
    # timestamps, cam_params = [], []
    cam_params = OrderedDict()

    with open(cam_file, 'r') as f:
        for line_ind, line in enumerate(f.readlines()):
            if line_ind == 0:
                # first line is youtube url
                continue
            entry = [float(x) for x in line.split()]
            timestamp = int(entry[0])
            K = np.array(entry[1:5], dtype=np.float32)  # (4,)  # normalized
            pose = np.array(entry[7:], dtype=np.float32).reshape(3, 4)
            assert timestamp not in cam_params
            cam_params[timestamp] = (K, pose)
            #
            # timestamps.append(timestamp)
            # cam_params.append((K, pose))
    if list(sorted(cam_params.keys())) != list(cam_params.keys()):
        print('[ERROR] cam file has timestamps not sroted', cam_file)
    # sorted_ids = np.argsort(timestamps)
    # cam_params = list(np.array(cam_params)[sorted_ids])
    # timestamps = list(np.array(timestamps)[sorted_ids])
    # assert (np.array(timestamps) == sorted(timestamps)).all()

    # return timestamps, cam_params
    return cam_params


class Dataset(MyBaseDataset):
    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference, is_test)

        if self.data_info.split == 'ibrnet_subset':
            # self.root_dir = '/svl/u/yzzhang/datasets/estate'
            self.root_dir = '/scr-ssd/yzzhang/estate'
        else:
            self.root_dir = '/viscam/data/estate'

        self.input_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ) if self.normalize else torchvision.transforms.Lambda(lambda t: t),
            ]
        )

        if self.data_info.cache_file != '':
            cache_file = self.data_info.cache_file
            master_only_print('loading metadata from cache', cache_file)
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            # self.clip_to_timestamps = cache['clip_to_timestamps']
            # self.clip_to_cam_params = cache['clip_to_cam_params']
            self.clip_to_image_ids_sorted_by_timestamps = cache['clip_to_image_ids_sorted_by_timestamps']
            self.clip_to_colmap_cameras = cache['clip_to_colmap_cameras']
            self.clip_to_colmap_images = cache['clip_to_colmap_images']
            self.clip_to_colmap_points3D = cache['clip_to_colmap_points3D']
            self.files = cache['files']
            return

        if False:
            all_clips = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root_dir, self.data_info.split, 'cameras'))]
            print(f"[INFO] total number of clips in split {self.data_info.split}: {len(all_clips)}")

            new_clips = []
            for clip in all_clips:
                f = os.path.join(DATASET_DIR, 'estate', self.data_info.split, f"{clip}")
                if os.path.exists(f):
                    new_clips.append(clip)
            all_clips = new_clips
            print(f"[INFO] total number of clips downloaded: {len(all_clips)}")
        else:
            all_clips = get_downloaded_clips(self.root_dir, self.data_info.split)

        # if not self.is_test:
        #     print('[DEBUG] temporarily take the first 500 clips')
        #     all_clips = all_clips[:500]  # FIXME

        # filter out clips that are incorrectly extracted

        if self.is_test:
            all_lines = np.loadtxt(os.path.join(ROOT, 'imaginaire/third_party/synsin/data/files/realestate.txt'),
                                   dtype=np.str)
            print('[INFO] number of image pairs in synsin test dataset', len(all_lines))

            # if True:  # FIXME: DEBUG
            #     all_lines = all_lines[:1000]

            filtered_lines: Dict[str, List[str]] = dict()
            for line in all_lines:
                clip = line[0]
                if clip not in all_clips:
                    print(f'[ERROR] clip unavailable {clip}')
                    # raise RuntimeError()
                    continue
                if clip not in filtered_lines:
                    filtered_lines[clip] = []
                filtered_lines[clip].append(line)
            all_lines = filtered_lines

            all_clips = all_lines.keys()

            print('[INFO] number of image pairs after filtering out failed ones', sum(len(v) for v in filtered_lines.values()))

        self.clip_to_image_ids_sorted_by_timestamps = dict()
        # self.clip_to_timestamps = dict()
        # self.clip_to_cam_params = dict()
        self.clip_to_colmap_cameras = dict()
        self.clip_to_colmap_images = dict()
        self.clip_to_colmap_points3D = dict()

        if os.getenv('DEBUG') == '1':
            print('[DEBUG] only take 50 clips')
            all_clips = list(all_clips)[:50]

        for clip in tqdm(all_clips):
            if False:
                cam_file = os.path.join(self.root_dir, self.data_info.split, "cameras", f"{clip}.txt")
                cam_params = get_cam_params_for_clip(cam_file)

                if len(cam_params) < self.sequence_length_max:
                    print('[ERROR] not enough number of frames in the clip')
                    continue

                if len(os.listdir(os.path.join(self.root_dir, self.data_info.split, "frames", clip))) != len(cam_params):
                    print('[ERROR] ffmpeg images extracted < images recorded', len(cam_params))
                    # FIXME: why did this happen?? re-download the data
                    # raise RuntimeError()
                    continue

            """ check colmap per-clip results """

            colmap_db = os.path.join(self.root_dir, self.data_info.split, "colmap", clip, "sparse")
            try:
                colmap_cameras, colmap_images, colmap_points3D = colmap_utils.read_model(colmap_db, ext=".bin")
            except Exception as e:
                print('[ERROR] cannot read from colmap output', clip, e)
                continue

            assert len(colmap_cameras) == 1
            if False:
                assert len(colmap_images) == len(cam_params)

                timestamps = list(sorted(cam_params.keys()))  # must be sorted!

            """ check colmap per-timestamp results """

            image_id_to_timestamps = dict()
            success = True
            for image_id, image in colmap_images.items():
                image_id_to_timestamps[image_id] = int(image.name.removesuffix('.png'))
            # for timestamp_ind in range(len(cam_params)):
                if False and colmap_images[timestamp_ind + 1].name != f"{timestamps[timestamp_ind]}.png":
                    # FIXME: raise error
                    # print(colmap_images[timestamp_ind + 1].name, timestamps[timestamp_ind])
                    #
                    # print(list(sorted(colmap_images.keys())))
                    # print([colmap_images[k].name for k in sorted(colmap_images.keys())])
                    # print(timestamps)
                    # raise RuntimeError
                    print(f"[ERROR] colmap timestamps do not correspond to data raw", clip)
                    if os.getenv('DEBUG') == '1':
                        names1 = [x.name for x in colmap_images.values()]
                        names2 = [f"{x}.png" for x in timestamps]
                        print(set(names1) - set(names2))
                        print(set(names2) - set(names1))
                        # for x1, x2 in zip(names1, names2):
                        #     if x1 != x2:
                        #         print(x1, x2)
                        # print(os.path.exists(os.path.join('/viscam/data/estate', self.data_info.split, "frames", clip, colmap_images[timestamp_ind + 1].name)))
                        # print(os.path.exists(os.path.join('/viscam/data/estate', self.data_info.split, "colmap", clip, 'images', colmap_images[timestamp_ind + 1].name)))
                        # print(os.path.exists(os.path.join('/viscam/data/estate', self.data_info.split, "frames", clip, f"{timestamps[timestamp_ind]}.png")))
                        # print(os.path.exists(os.path.join('/viscam/data/estate', self.data_info.split, "colmap", clip, 'images', f"{timestamps[timestamp_ind]}.png")))
                    success = False
                    break

                num_points = len([p for p in image.point3D_ids if p != -1])
                if num_points < 250:  # FIXME: ?
                    print(f"[ERROR] colmap < 250 points", num_points, len(image.point3D_ids))
                    success = False
                    break

            if not success:
                continue
            # else:
            #     timestamps = list(sorted(cam_params.keys()))  # must be sorted!
            #     colmap_cameras, colmap_images, colmap_points3D = None, None, None

            image_ids_sorted_by_timestamps = [image_id for image_id, _ in
                                              sorted(image_id_to_timestamps.items(), key=lambda item: item[1])]  # is it correct

            # self.clip_to_timestamps[clip] = timestamps
            # self.clip_to_cam_params[clip] = cam_params
            self.clip_to_image_ids_sorted_by_timestamps[clip] = image_ids_sorted_by_timestamps
            self.clip_to_colmap_cameras[clip] = colmap_cameras
            self.clip_to_colmap_images[clip] = colmap_images
            self.clip_to_colmap_points3D[clip] = colmap_points3D

        all_clips = self.clip_to_colmap_cameras.keys()#= self.clip_to_timestamps.keys()
        print('[INFO] number of clips after checking camera params and colmap', len(all_clips))

        files: List[Tuple[str, Union[int, None], Union[int, None]]] = []
        for clip in tqdm(all_clips):
            # if sequence_length = 4, t = 0 or 1 is not allowed
            if self.is_test:  # TODO
                timestamps = self.clip_to_timestamps[clip]
                for line in all_lines[clip]:
                    # otherwise locate the index of source in timestamps_this_clip
                    src = timestamps.index(int(line[1]))
                    if not self.sequence_length_max - 2 <= src < len(timestamps) - 1:
                        continue
                    qry = timestamps.index(int(line[2]))
                    files.append((clip, src, qry))
            else:
                # sample timestamp for query on the fly
                files.extend([(clip, src, None) for src in range(self.sequence_length_max - 2, len(self.clip_to_image_ids_sorted_by_timestamps[clip]) - 1)])
                # files.append((clip, None, None))  # FIXME: hack, so that data loading is faster

        self.files = files

        cache = {
            # 'clip_to_timestamps': self.clip_to_timestamps,
            # 'clip_to_cam_params': self.clip_to_cam_params,
            'clip_to_image_ids_sorted_by_timestamps': self.clip_to_image_ids_sorted_by_timestamps,
            'clip_to_colmap_cameras': self.clip_to_colmap_cameras,
            'clip_to_colmap_images': self.clip_to_colmap_images,
            'clip_to_colmap_points3D': self.clip_to_colmap_points3D,
            'files': files,
        }
        cache_file = os.path.join(
            DATASET_DIR, 'estate', f"metadata_{cfg.data.name}_{self.data_info.split}_{time.time()}.pkl")
        print(f"[INFO] saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def __len__(self):
        return len(self.files)

    def get_data_for(self, clip, image_id):
        """

        Args:
            clip_id:
            id: index in the clip sorted by timestamps

        Returns:

        """
        data = dict()
        img_item = self.clip_to_colmap_images[clip][image_id]
        # timestamp = self.clip_to_timestamps[clip][timestamp_ind]
        # K, pose = self.clip_to_cam_params[clip][timestamp]
        # data['camera'] = K
        # data['pose'] = pose
        data['timestamp'] = int(img_item.name.removesuffix('.png'))

        rgb_file = os.path.join(self.root_dir, self.data_info.split, "frames", clip, img_item.name)
        with Image.open(rgb_file) as rgb:
            if False:
                assert rgb.size == (256, 256)
            rgb = self.input_transform(rgb)
            data['image'] = rgb
            # data['image_path'] = rgb_file
        # if self.clip_to_colmap_cameras[clip] is None:
        #     return data

        # print([self.clip_to_colmap_images[clip][k].name for k in sorted(self.clip_to_colmap_images[clip].keys())])
        # if os.getenv('DEBUG') == '1':
        # assert img_item.name == str(timestamp) + '.png'
        cam = self.clip_to_colmap_cameras[clip][img_item.camera_id]
        points3D = self.clip_to_colmap_points3D[clip]

        # if not os.path.exists(rgb):
        #     print('trying to get index', id, 'from', len(self.clip_to_timestamps[clip]), clip)
        #     print('number of timestamps extracted', len(os.listdir(os.path.dirname(rgb))))
        #     raise RuntimeError()
        if False:
            assert (cam.width, cam.height) == (256, 256)
        # return rgb, K, pose

        # load sparse point cloud
        qvec = img_item.qvec
        tvec = img_item.tvec
        xyzs = [(point3D_id, xy, points3D[point3D_id].xyz)
                for xy, point3D_id in zip(img_item.xys, img_item.point3D_ids)
                if point3D_id != -1]

        if os.getenv('DEBUG') == '1':
            assert len(xyzs) >= 8, len(xyzs)  # visible_points_count
            assert len(img_item.xys) == len(img_item.point3D_ids)
            assert len(cam.params) == 4  # pinhole
        # the following is from nerf dataset, they probably use simple pinhole
        # https://colmap.github.io/format.html
        # colmap_camera = np.array([cam.params[0] / cam.width, cam.params[0] / cam.height,
        #                      cam.params[1] / cam.width, cam.params[2] / cam.height], dtype=np.float32) # (4,)
        colmap_camera = np.array([cam.params[0] / cam.width, cam.params[1] / cam.height,
                                  cam.params[2] / cam.width, cam.params[3] / cam.height], dtype=np.float32)  # (4,)
        R_cam_world = colmap_utils.qvec2rotmat(qvec).astype(np.float32)
        T_cam_world = np.array(tvec, dtype=np.float32).reshape(3, 1)
        colmap_pose = np.hstack((R_cam_world, T_cam_world))  # (3, 4)

        if False and os.getenv('DEBUG') == '1':
            if not np.allclose(colmap_pose, pose, atol=1e-4):
                print('[ERROR] colmap extrinsics does not match raw data')
                import ipdb; ipdb.set_trace()
            if not np.allclose(colmap_camera, K, atol=1e-4):
                print('[ERROR] colmap intrinsics does not match raw data')
                import ipdb; ipdb.set_trace()

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

        mask = data['xyzs'][2, :] >= 1e-3  # depth must be positive
        if os.getenv('DEBUG') == '1':
            if not np.all(mask):
                print("[INFO] sparse point cloud depth > 1e-3 mean", np.mean(mask))
        data['xys_cam'] = data['xys_cam'][:, mask]
        data['xyzs'] = data['xyzs'][:, mask]

        if False and os.getenv('DEBUG') == '1':
            # if np.quantile(xyzs_cam_homo[2, :], q=0.3) < 1:
            #     print('[INFO] colmap 30% points has depth < 1',
            #           np.quantile(xyzs_cam_homo[2, :], q=0.3),
            #           np.quantile(xyzs_cam_homo[2, :], q=0.5),
            #           )

            # print('[INFO] colmap min depth', xyzs_cam_homo[2, :].min())
            print('[INFO] colmap 10% depth', np.quantile(xyzs_cam_homo[2, :], q=0.1))
            print('[INFO] colmap 90% depth', np.quantile(xyzs_cam_homo[2, :], q=0.9))
            print('[INFO] colmap max depth', xyzs_cam_homo[2, :].max())

        if os.getenv('DEBUG') == '1':
            # from imaginaire.third_party.geom_free.data.realestate import load_sparse_model_example
            # ex = load_sparse_model_example(os.path.join(DATASET_DIR, 'estate_sfm', self.data_info.split, clip),
            #                                f"{timestamp}.png", f"{timestamp}.png", size=self.img_size)
            downsample_ratio_x = cam.width / self.img_size[1]
            downsample_ratio_y = cam.height / self.img_size[0]
            K = np.array([[cam.params[0], 0.0, cam.params[2]],
                          [0.0, cam.params[1], cam.params[3]],
                          [0.0, 0.0, 1.0]])

            xys = img_item.xys
            p3D = img_item.point3D_ids
            pmask = p3D > 0
            # if verbose: print("Found {} 3d points in sparse model.".format(pmask.sum()))
            xys = xys[pmask]
            p3D = p3D[pmask]
            worlds = np.stack([points3D[id_].xyz for id_ in p3D]) # N, 3
            # project to current view
            worlds = worlds[..., None] # N,3,1
            pixels = K[None,...]@(R_cam_world[None,...]@worlds+T_cam_world.reshape(3,)[None,...,None])
            pixels = pixels.squeeze(-1) # N,3

            # instead of using provided xys, one could also project pixels, ie
            # xys ~ pixels[:,:2]/pixels[:,[2]]
            if False:
                points = np.concatenate([xys, pixels[:,[2]]], axis=1)
            else:
                points = np.concatenate([pixels[:, :2] / pixels[:, [2]], pixels[:, [2]]], axis=1)
            sparse_points = points

            # code to convert to sparse depth map
            # xys = points[:,:2]
            # xys = np.round(xys).astype(np.int)
            # xys[:,0] = xys[:,0].clip(min=0,max=w-1)
            # xys[:,1] = xys[:,1].clip(min=0,max=h-1)
            # indices = xys[:,1]*w+xys[:,0]
            # flatdm = np.zeros(h*w)
            # flatz = pixels[:,2]
            # np.put_along_axis(flatdm, indices, flatz, axis=0)
            # sparse_dm = flatdm.reshape(h,w)

            # resize
            K[0, :] = K[0, :] / downsample_ratio_x
            K[1, :] = K[1, :] / downsample_ratio_y
            ## points
            points[:, 0] = points[:, 0] / downsample_ratio_x
            points[:, 1] = points[:, 1] / downsample_ratio_y

            # print('[INFO] colmap proj min depth', points[:, 2].min())
            # print('[INFO] colmap proj 10% depth', np.quantile(points[:, 2], q=0.1))
            # print('[INFO] colmap proj 90% depth', np.quantile(points[:, 2], q=0.9))
            # print('[INFO] colmap proj max depth', points[:, 2].max())

            points = points[mask, :]
            assert np.allclose(xyzs_cam_homo[2, mask], points[:, 2], atol=1e-3), (xyzs_cam_homo[2, :10], points[:10, 2])

            # project to image plane
            src_pt3d_pxpy = K @ data['xyzs']  # (3, 3) @ (3, n) -> (3, n)
            src_pt3d_pxpy = src_pt3d_pxpy[0:2, :] / src_pt3d_pxpy[2:, :]  # (2, n)
            max_pixel_diff = np.abs(src_pt3d_pxpy.round() - points[:, :2].T.round()).max()
            if max_pixel_diff > 0:
                print('[ERROR] pxpy diff > 0', max_pixel_diff)
            assert np.allclose(data['xyzs'][2, :], points[:, 2], atol=1e-3), (data['xyzs'][2, :10], points[:10, 2])  # TODO

        if os.getenv('DEBUG') == '1':
            # info = NeRFDataset._info_transform(
            #     {"img": rgb, "qvec": qvec, "tvec": tvec, "xyzs": xyzs,
            #      "camera_params": cam.params},
            #     # (w * img_pre_downsample_ratio / self.img_w,
            #     #  h * img_pre_downsample_ratio / self.img_h)
            #     # (1280 / self.img_size[1], 720 / self.img_size[0])  # FIXME: not sure if the original res is 720 x 1280
            #     (cam.width / self.img_size[1], cam.height / self.img_size[0])  # FIXME: not sure
            # )
            #
            # if not np.allclose(data['xyzs'], info['xyzs'], atol=1e-4):
            #     import ipdb; ipdb.set_trace()

            # Compute K matrix
            f_x = cam.params[0] / downsample_ratio_x
            f_y = cam.params[1] / downsample_ratio_y
            p_x = cam.params[2] / downsample_ratio_x
            p_y = cam.params[3] / downsample_ratio_y
            K = np.array([
                [f_x, 0, p_x],
                [0, f_y, p_y],
                [0, 0, 1]
            ], dtype=np.float32)
            # Scale coordinates of tracked points, then compute and normalize the depth for each point
            I_Zero = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=np.float32)
            P = K @ I_Zero @ G_cam_world
            m_det_sign = np.sign(np.linalg.det(P[:, :-1]))
            m3_norm = np.linalg.norm(P[2][:-1])

            # Convert xyzs_world to homogeneous coordinates
            _, xys_cam, _ = zip(*xyzs)
            xys_cam = np.array(xys_cam).T.astype(np.float32)
            xys_cam[0] /= downsample_ratio_x
            xys_cam[1] /= downsample_ratio_y

            # Reproject to image plane to obtain depths
            xys_cam_reproj = K @ I_Zero @ xyzs_cam_homo  # (4, n)
            depths = (m_det_sign * xys_cam_reproj[-1]) / m3_norm
            assert m_det_sign == 1, m_det_sign
            xys_cam_reproj /= xys_cam_reproj[-1]

            # print(xys_cam[:, :10])
            # print(xys_cam_reproj[:2, :10])

            xys_cam_reproj = xys_cam_reproj[:, mask]

            # taken from stereo.py
            src_pt3d_pxpy = K @ data['xyzs']  # 3x3 * 3xN_pt -> 3xN_pt
            src_pt3d_pxpy = src_pt3d_pxpy[0:2, :] / src_pt3d_pxpy[2:, :]  # 2xN_pt
            try:
                assert np.allclose(xys_cam_reproj[:2, :], src_pt3d_pxpy), (xys_cam_reproj[:2, :] - src_pt3d_pxpy).abs().max()
            except:
                import ipdb; ipdb.set_trace()
                print()

        if False and os.getenv('DEBUG') == '1':
            if xyzs.shape[1] > 400:
                from tu.loggers.html_table import HTMLTableVisualizer
                from tu.loggers.html_helper import BaseHTMLHelper
                import matplotlib.pyplot as plt
                from mpl_toolkits import mplot3d

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                pc = xyzs.copy()
                mean_z = pc[2, :].mean()
                std_z = pc[2, :].std()
                pc = pc[:, abs(pc[2, :] - mean_z) < 3 * std_z]
                ax.scatter(pc[0, :], pc[1, :], pc[2, :], s=0.01)

                vis = HTMLTableVisualizer('/viscam/u/yzzhang/assets/pointcloud', 'estate')
                BaseHTMLHelper.print_url(vis)
                import ipdb; ipdb.set_trace()
                with vis.html():
                    BaseHTMLHelper.dump_table(vis, layout=[[plt]], table_name='', col_type='figure')

        # data['camera'] = colmap_camera
        # data['pose'] = colmap_pose # info['G_cam_world'][:3, :]
        return data

    def __getitem__(self, index):
        # Then load the image and generate that
        clip, src, qry = self.files[index]
        if False and src is None:
            src = np.random.choice(range(self.sequence_length - 2, len(self.clip_to_timestamps[clip]) - 1))
        # else:
        #     # don't do this to ensure test split is the same as synsin
        #     src = src + np.random.choice(self.sequence_length_max - self.sequence_length + 1)
        if qry is None:
            qry = select_id_query_stereo_mag(len(self.clip_to_image_ids_sorted_by_timestamps[clip]), src)

        # collect timestamps
        # sequence_data = list(map(lambda ind: self.get_data_for(clip, ind), range(src + 2 - self.sequence_length, src + 2)))
        # query_data = self.get_data_for(clip, qry)  # assume timestamps sorted

        image_ids = [self.clip_to_image_ids_sorted_by_timestamps[clip][ind] for ind in range(src + 2 - self.sequence_length, src + 2)]
        qry_image_id = self.clip_to_image_ids_sorted_by_timestamps[clip][qry]

        sequence_data = [self.get_data_for(clip=clip, image_id=image_id) for image_id in image_ids]
        timestamps = [d['timestamp'] for d in sequence_data]
        if list(sorted(timestamps)) != timestamps:
            print("[ERROR]", timestamps, src)
            import ipdb; ipdb.set_trace()

        query_data = self.get_data_for(clip=clip, image_id=qry_image_id)

        data = dict()
        # data['image_path'] = [d['image_path'] for d in sequence_data]
        data['images'] = torch.stack([d['image'] for d in sequence_data])
        data['cameras'] = torch.from_numpy(np.stack([d['camera'] for d in sequence_data]))
        data['poses'] = torch.from_numpy(np.stack([d['pose'] for d in sequence_data]))
        # if 'xyzs' in d:
        data['xyzs']: List = [torch.from_numpy(d['xyzs']) for d in sequence_data]
        data['xys_cam']: List = [torch.from_numpy(d['xys_cam']) for d in sequence_data]

        data['query_image'] = query_data['image']
        data['query_camera'] = torch.from_numpy(query_data['camera'])
        data['query_pose'] = torch.from_numpy(query_data['pose'])
        data['query_xyzs'] = torch.from_numpy(query_data['xyzs'])
        data['query_xys_cam'] = torch.from_numpy(query_data['xys_cam'])

        # assert clip_id == file_name[0]
        # assert str(self.clip_id_to_timestamps[clip_id][id_src]) == file_name[1]
        # assert str(self.clip_id_to_timestamps[clip_id][id_qry]) == file_name[2]

        # # synsin process
        # src_image_name = os.path.join(
        #     self.root_dir, self.data_info.split, file_name[0], f"{file_name[1]}.png",   # can convert to png
        # )
        # tgt_image_name = os.path.join(
        #     self.root_dir, self.data_info.split, file_name[0], f"{file_name[2]}.png",   # can convert to png
        # )
        # intrinsics = file_name[3:7].astype(np.float32) / float(256)  # FIXME: is it 256?
        # print(intrinsics)
        # src_pose = file_name[7:19].astype(np.float32).reshape(3, 4)
        # tgt_pose = file_name[19:].astype(np.float32).reshape(3, 4)
        #
        # import ipdb; ipdb.set_trace()
        # src_image = self.input_transform(Image.open(src_image_name))
        # tgt_image = self.input_transform(Image.open(tgt_image_name))
        #
        # src_image_prev = self.input_transform(Image.open)

        return data


def collate_fn(batch):
    new_batch = dict()
    for k in batch[0].keys():
        new_batch[k] = [data[k] for data in batch]
        if k not in ['xyzs', 'xys_cam', 'query_xyzs', 'query_xys_cam', 'image_path', 'depths',
                     'image_paths', 'query_image_path', 'meta_info']:
            new_batch[k] = default_collate(new_batch[k])
    return new_batch


if __name__ == "__main__":
    from imaginaire.config import get_attr_dict
    cfg = get_attr_dict(os.path.join(ROOT, 'configs/datasets/estate_synsin_eval.yaml'))
    print(cfg)

    d = Dataset(cfg, is_test=False)

    # TODO: comment back
    succ_count = 0
    fail_count = 0
    for i in range(len(d)):
        try:
            data = d[i]
            succ_count += 1
        except Exception as e:
            fail_count += 1
            print(e)
            raise(e)
            # break
    print('succ', succ_count, 'fail', fail_count)

    print(d[0].keys())

    from torch.utils.data import DataLoader

    dl = DataLoader(d, batch_size=4, shuffle=False,
                    drop_last=True, num_workers=0,
                    collate_fn=collate_fn)

    for batch in dl:
        print(batch.keys())
        break
