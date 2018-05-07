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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}
