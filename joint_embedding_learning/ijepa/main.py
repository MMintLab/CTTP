# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from joint_embedding_learning.ijepa.utils.distributed import init_distributed
from joint_embedding_learning.ijepa.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which devices to use on local machine')
    


if __name__ == '__main__':
    args = parser.parse_args()

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    fname = args.fname
    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    app_main(args=params, device=args.device)

    
