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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pickle
import os
import sys
import yaml

from core.config import cfg
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
import core.config
import utils.logging


def parse_args():
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument(
        'output_dir', nargs=1, help='results directory', type=str
    )
    parser.add_argument(
        '--dataset',
        dest='dataset_name',
        help='dataset to re-evaluate',
        default='voc_2007_test',
        type=str
    )
    parser.add_argument(
        '--matlab',
        dest='matlab_eval',
        help='use matlab for evaluation',
        action='store_true'
    )
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def do_reval(dataset_name, output_dir, args):
    dataset = JsonDataset(dataset_name)
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = pickle.load(f)
    # Override config with the one saved in the detections file
    if args.cfg_file is not None:
        core.config.merge_cfg_from_cfg(yaml.load(dets['cfg']))
    else:
        core.config._merge_a_into_b(yaml.load(dets['cfg']), cfg)
    results = task_evaluation.evaluate_all(
        dataset,
        dets['all_boxes'],
        dets['all_segms'],
        dets['all_keyps'],
        output_dir,
        use_matlab=args.matlab_eval
    )
    task_evaluation.log_copy_paste_friendly_results(results)


if __name__ == '__main__':
    utils.logging.setup_logging(__name__)
    args = parse_args()
    if args.comp_mode:
        cfg.TEST.COMPETITION_MODE = True
    output_dir = os.path.abspath(args.output_dir[0])
    do_reval(args.dataset_name, output_dir, args)
