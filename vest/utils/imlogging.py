# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import datetime
import os

from vest.utils.distributed import master_only
from vest.utils.distributed import master_only_print as print
from vest.utils.meters import set_summary_writer


def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(config_path, logdir, log_prefix=None):
    r"""Create log directory for storing checkpoints and output images.

    Args:
        config_path (str): Path to the configuration file.
        logdir (str): Log directory name
    Returns:
        str: Return log dir
    """
    config_file = os.path.basename(config_path)
    root_dir = 'logs'
    date_uid = get_date_uid()
    # example: logs/2019_0125_1047_58_spade_cocostuff
    log_file = '_'.join([date_uid, os.path.splitext(config_file)[0]])
    if log_prefix is not None:
        log_file = f"{log_prefix}_{log_file}"
    if logdir is None:
        logdir = os.path.join(root_dir, log_file)
    return date_uid, logdir


def my_init_logging(log_prefix, root_dir='logs', seed=0):
    date_uid = get_date_uid()
    # example: logs/{log_prefix}_{date}_{seed}
    log_file = '_'.join([log_prefix, date_uid, f"seed_{seed}"])
    if log_prefix == '':
        log_file = log_file[1:]  # remove '_'
    logdir = os.path.join(root_dir, log_file)
    logdir = os.path.abspath(logdir)
    return date_uid, logdir


@master_only
def make_logging_dir(logdir):
    r"""Create the logging directory

    Args:
        logdir (str): Log directory name
    """
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    set_summary_writer(tensorboard_dir)
