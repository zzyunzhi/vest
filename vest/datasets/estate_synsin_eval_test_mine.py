import numpy as np
import os
import json
from vest.datasets.estate_synsin_eval_v2 import Dataset as EstateDataset


class Dataset(EstateDataset):
    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference=is_inference, is_test=is_test)

        available_clips = self.clip_to_image_ids_sorted_by_timestamps.keys()
        self.files = []
        count_fail = 0
        assert os.path.exists('/viscam/data/estate/raw/mine_test_pairs.json')
        with open('/viscam/data/estate/raw/mine_test_pairs.json', 'r') as f:
            for line in f.readlines():
                pair = json.loads(line)
                clip = pair['sequence_id']
                if clip in available_clips:
                    src = pair['src_img_obj']['frame_ts']
                    if self.sample_strategy == 'random':
                        tgt = pair['tgt_img_obj_random']['frame_ts']
                    elif self.sample_strategy == 'random_5':
                        tgt = pair['tgt_img_obj_5_frames']['frame_ts']
                    elif self.sample_strategy == 'random_10':
                        tgt = pair['tgt_img_obj_10_frames']['frame_ts']

                    line = f"{clip} {src} {tgt}"
                    self.files.append(line)
                else:
                    count_fail += 1
        print('[INFO] succesfully found clips', len(self.files), 'failed count', count_fail)
