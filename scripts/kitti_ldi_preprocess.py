#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for preprocessing KITTI data.
"""
import fnmatch
import os

from vest.datasets.kitti_ldi import raw_city_sequences
from common import DATASET_DIR, ROOT


def main():
    kitti_data_root = os.path.join(DATASET_DIR, 'kitti_toy')
    print('data_root', kitti_data_root)
    spss_exec = os.path.join(ROOT, 'vest/third_party/spsstereo_git_patch/spsstereo')
    spss_exec = '/viscam/u/yzzhang/projects/'

    root_dir = os.path.join(kitti_data_root, 'kitti_raw')
    # exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.png'
    seq_names = raw_city_sequences()
    img_list_src = []
    folder_list_spss = []

    for seq_id in seq_names:
        seq_date = seq_id[0:10]
        seq_dir = os.path.join(root_dir, seq_date, '{}_sync'.format(seq_id))
        for root, _, filenames in os.walk(os.path.join(seq_dir, 'image_02')):
            for filename in fnmatch.filter(filenames, '*.png'):
                src_img_name = os.path.join(root, filename)
                if True:  # exclude_img not in src_img_name:
                    img_list_src.append(src_img_name)
                    folder_list_spss.append(
                        os.path.join(root_dir, 'spss_stereo_results',
                                     src_img_name.split('/')[-4]))

    img_list_trg = [f.replace('image_02', 'image_03') for f in img_list_src]
    for ix, (src_im, trg_im, dst) in enumerate(
            zip(img_list_src, img_list_trg, folder_list_spss)):
        if ix % 50 == 0:
            print('{}/{}'.format(ix, len(img_list_src)))
        # if not os.path.exists(dst):
        #     os.makedirs(dst)
        os.makedirs(dst, exist_ok=True)
        os.system('{} {} {}'.format(spss_exec, src_im, trg_im))
        os.system('mv ./*.png {}/'.format(dst))
        os.system('mv ./*.txt {}/'.format(dst))


if __name__ == '__main__':
    main()
