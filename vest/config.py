# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""Config utilities for yml file."""

import collections
import functools
import os
import re
import json
from common import ROOT

import yaml
from vest.utils.distributed import master_only_print as print


class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()
        # Set default parameters.
        # Logging.
        large_number = 1000000000
        self.snapshot_save_iter = large_number
        self.snapshot_save_epoch = large_number
        self.snapshot_save_start_iter = 0
        self.snapshot_save_start_epoch = 0
        self.image_save_iter = large_number
        self.image_display_iter = large_number
        self.max_epoch = large_number
        self.max_iter = large_number
        self.logging_iter = 100

        # Trainer.
        self.trainer = AttrDict(
            model_average=False,
            model_average_beta=0.9999,
            model_average_start_iteration=1000,
            model_average_batch_norm_estimation_iteration=30,
            model_average_remove_sn=True,
            image_to_tensorboard=False,
            hparam_to_tensorboard=False,
            distributed_data_parallel='pytorch',
            delay_allreduce=True,
            gan_relativistic=False,
            gen_step=1,
            dis_step=1)

        # Networks.
        self.gen = AttrDict(type='vest.generators.dummy')
        self.dis = AttrDict(type='vest.discriminators.dummy')

        # Optimizers.
        self.gen_opt = AttrDict(type='adam',
                                fused_opt=True,
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        self.dis_opt = AttrDict(type='adam',
                                fused_opt=True,
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        # Data.
        self.data = AttrDict(name='dummy',
                             type='vest.datasets.images',
                             num_workers=0)
        self.test_data = AttrDict(name='dummy',
                                  type='vest.datasets.images',
                                  num_workers=0,
                                  test=AttrDict(is_lmdb=False,
                                                roots='',
                                                batch_size=1))


# Cudnn.
        self.cudnn = AttrDict(deterministic=False,
                              benchmark=True)

        # Others.
        self.pretrained_weight = ''
        self.inference_args = AttrDict()

        if isinstance(filename, dict):
            cfg_dict = filename
        else:
            # Update with given configurations.
            assert os.path.exists(filename), 'File {} not exist.'.format(filename)
            try:
                cfg_dict = load_cfg_to_dict(filename)
            except EnvironmentError:
                print('Please check the file with name of "%s"', filename)
        recursive_update(self, cfg_dict)

        # Put common opts in both gen and dis.
        if 'common' in cfg_dict:
            self.common = AttrDict(**cfg_dict['common'])
            self.gen.common = self.common
            self.dis.common = self.common

        if verbose:
            print(' vest config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def load_cfg_to_dict(filename):
    assert os.path.exists(filename), 'File {} not exist.'.format(filename)
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(filename, 'r') as f:
        cfg_dict = yaml.load(f, Loader=loader)
    return cfg_dict


def rsetattr(obj, attr, val):
    """Recursively find object and set value"""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find object and return value"""

    def _getattr(obj, attr):
        r"""Get attribute."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u, strict=False):
    """Recursively update AttrDict d with AttrDict u"""
    # if strict, requires all keys in u to appear in d
    for key, value in u.items():
        if strict and key not in d.__dict__:
            print(f"{key} exists in u but not in d")
            raise
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d

#
# def get_config_dict(filename: str, depth=0) -> AttrDict:
#     if depth > 5:
#         raise RuntimeError("might be an infinite loop!")
#
#     print(f"loading from {filename}")
#
#     cfg_dict = load_cfg_to_dict(filename)
#     if cfg_dict['config_base'] != "":
#         # print extra configs
#         print(json.dumps(cfg_dict, sort_keys=True, indent=4))
#
#         filename = os.path.expanduser(os.path.join(ROOT, cfg_dict['config_base']))
#         cfg_dict_out = get_config_dict(filename, depth=depth+1)
#
#         recursive_update(cfg_dict_out, cfg_dict, strict=True)
#     else:
#         cfg_dict_out = cfg_dict
#
#     print(cfg_dict_out.get('message', 'no message from this config file'))
#
#     return cfg_dict_out


def get_attr_dict(filename):
    cfg_dict = load_cfg_to_dict(filename)
    if cfg_dict['config_base']:
        cfg = get_attr_dict(cfg_dict['config_base'])
        recursive_update(cfg, cfg_dict)
    else:
        cfg = cfg_dict

    return AttrDict(cfg)


def get_config_recursive(filename):
    filename = os.path.expanduser(os.path.join(ROOT, filename))
    print('reading configs from', filename)

    cfg = load_cfg_to_dict(filename)
    while cfg['config_base']:
        print('recursing...')
        base_filename = cfg.pop('config_base')
        if base_filename == 'default':
            base_filename = os.path.join(os.path.dirname(filename), 'base.yaml')
        if filename == base_filename:
            raise RuntimeError('infinite loop', filename)
        base_cfg = get_config_recursive(base_filename)
        recursive_update(base_cfg, cfg, strict=True)
        cfg = base_cfg

    return Config(filename=cfg)


def get_config(filename):
    cfg_dict = load_cfg_to_dict(filename)
    if cfg_dict['config_base']:
        print(json.dumps(cfg_dict, sort_keys=True, indent=4))

        if cfg_dict['config_base'] == 'default':
            cfg = Config(os.path.join(os.path.dirname(filename), 'base.yaml'))
        else:
            cfg = Config(os.path.expanduser(os.path.join(ROOT, cfg_dict['config_base'])))
        recursive_update(cfg, cfg_dict, strict=True)
    else:
        cfg = Config(filename)

    print(getattr(cfg, 'message', 'no message from this config file'))

    return cfg