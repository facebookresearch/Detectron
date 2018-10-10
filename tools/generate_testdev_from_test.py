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

"""Given a full set of results (boxes, masks, or keypoints) on the 2017 COCO
test set, this script extracts the results subset that corresponds to 2017
test-dev. The test-dev subset can then be submitted to the COCO evaluation
server.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys

from detectron.datasets.dataset_catalog import get_ann_fn
from detectron.utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json', dest='json_file',
        help='detections json file',
        default='', type=str)
    parser.add_argument(
        '--output-dir', dest='output_dir',
        help='output directory',
        default='/tmp', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def convert(json_file, output_dir):
    print('Reading: {}'.format(json_file))
    with open(json_file, 'r') as fid:
        dt = json.load(fid)
    print('done!')

    test_image_info = get_ann_fn('coco_2017_test')
    with open(test_image_info, 'r') as fid:
        info_test = json.load(fid)
    image_test = info_test['images']
    image_test_id = [i['id'] for i in image_test]
    print('{} has {} images'.format(test_image_info, len(image_test_id)))

    test_dev_image_info = get_ann_fn('coco_2017_test-dev')
    with open(test_dev_image_info, 'r') as fid:
        info_testdev = json.load(fid)
    image_testdev = info_testdev['images']
    image_testdev_id = [i['id'] for i in image_testdev]
    print('{} has {} images'.format(test_dev_image_info, len(image_testdev_id)))

    dt_testdev = []
    print('Filtering test-dev from test...')
    t = Timer()
    t.tic()
    for i in range(len(dt)):
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(dt)))
        if dt[i]['image_id'] in image_testdev_id:
            dt_testdev.append(dt[i])
    print('Done filtering ({:2}s)!'.format(t.toc()))

    filename, file_extension = os.path.splitext(os.path.basename(json_file))
    filename = filename + '_test-dev'
    filename = os.path.join(output_dir, filename + file_extension)
    with open(filename, 'w') as fid:
        info_test = json.dump(dt_testdev, fid)
    print('Done writing: {}!'.format(filename))


if __name__ == '__main__':
    opts = parse_args()
    convert(opts.json_file, opts.output_dir)
