from tu.ddp import master_only_print
import torch


class MyBaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_inference, is_test):
        super().__init__()
        # self.is_inference = is_inference
        # self.is_test = is_test
        if is_test:
            data_info = cfg.test_data.test
        else:
            if is_inference:
                data_info = cfg.data.val
            else:
                data_info = cfg.data.train
        self.data_info = data_info
        self.is_inference = is_inference
        self.is_test = is_test

        master_only_print("===== data info =======")
        master_only_print('is_inference', is_inference, 'is_test', is_test)
        master_only_print(self.data_info)
        master_only_print("=======================")

        self.sequence_length = cfg.data.train.initial_sequence_length
        self.sequence_length_max = cfg.data.train.max_sequence_length
        self.img_size = tuple(cfg.data.img_size)  # (h, w)
        self.normalize = cfg.data.input_types[0].images.normalize

        assert self.normalize
        assert self.sequence_length_max == self.sequence_length

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        # should use self.normalize in this function
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
