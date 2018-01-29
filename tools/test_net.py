#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from core.rpn_generator import generate_rpn_on_dataset
from core.rpn_generator import generate_rpn_on_range
from core.test_engine import test_net, test_net_on_dataset
from datasets import task_evaluation
import utils.c2
import utils.logging

utils.c2.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(ind_range=None, multi_gpu_testing=False):
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        child_func = generate_rpn_on_range
        parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        child_func = test_net
        parent_func = test_net_on_dataset

    is_parent = ind_range is None

    if is_parent:
        # Parent case:
        # In this case we're either running inference on the entire dataset in a
        # single process or (if multi_gpu_testing is True) using this process to
        # launch subprocesses that each run inference on a range of the dataset
        if len(cfg.TEST.DATASETS) == 0:
            cfg.TEST.DATASETS = (cfg.TEST.DATASET, )
            cfg.TEST.PROPOSAL_FILES = (cfg.TEST.PROPOSAL_FILE, )

        all_results = {}
        for i in range(len(cfg.TEST.DATASETS)):
            cfg.TEST.DATASET = cfg.TEST.DATASETS[i]
            if cfg.TEST.PRECOMPUTED_PROPOSALS:
                cfg.TEST.PROPOSAL_FILE = cfg.TEST.PROPOSAL_FILES[i]
            results = parent_func(multi_gpu=multi_gpu_testing)
            all_results.update(results)

        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)
    else:
        # Subprocess child case:
        # In this case test_net was called via subprocess.Popen to execute on a
        # range of inputs on a single dataset (i.e., use cfg.TEST.DATASET and
        # don't loop over cfg.TEST.DATASETS)
        child_func(ind_range=ind_range)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    main(ind_range=args.range, multi_gpu_testing=args.multi_gpu_testing)
