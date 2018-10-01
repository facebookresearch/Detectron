#!/usr/bin/env python

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

"""Script for visualizing results saved in a detections.pkl file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset',
        default='coco_2014_minival',
        type=str
    )
    parser.add_argument(
        '--detections',
        dest='detections',
        help='detections pkl file',
        default='',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='./tmp/vis-output',
        type=str
    )
    parser.add_argument(
        '--first',
        dest='first',
        help='only visualize the first k images',
        default=0,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def vis(dataset, detections_pkl, thresh, output_dir, limit=0):
    ds = JsonDataset(dataset)
    roidb = ds.get_roidb()

    dets = load_object(detections_pkl)

    assert all(k in dets for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections pkl file in the format used by test_engine.py'

    all_boxes = dets['all_boxes']
    all_segms = dets['all_segms']
    all_keyps = dets['all_keyps']

    def id_or_index(ix, val):
        if len(val) == 0:
            return val
        else:
            return val[ix]

    for ix, entry in enumerate(roidb):
        if limit > 0 and ix >= limit:
            break
        if ix % 10 == 0:
            print('{:d}/{:d}'.format(ix + 1, len(roidb)))

        im = cv2.imread(entry['image'])
        im_name = os.path.splitext(os.path.basename(entry['image']))[0]

        cls_boxes_i = [
            id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
        ]
        cls_segms_i = [
            id_or_index(ix, cls_k_segms) for cls_k_segms in all_segms
        ]
        cls_keyps_i = [
            id_or_index(ix, cls_k_keyps) for cls_k_keyps in all_keyps
        ]

        vis_utils.vis_one_image(
            im[:, :, ::-1],
            '{:d}_{:s}'.format(ix, im_name),
            os.path.join(output_dir, 'vis'),
            cls_boxes_i,
            segms=cls_segms_i,
            keypoints=cls_keyps_i,
            thresh=thresh,
            box_alpha=0.8,
            dataset=ds,
            show_class=True
        )


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.dataset,
        opts.detections,
        opts.thresh,
        opts.output_dir,
        limit=opts.first
    )
