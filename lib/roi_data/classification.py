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

"""Construct minibatches for Classification training. Handles the minibatch blobs
that are specific to Classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from core.config import cfg
import utils.blob as blob_utils
import utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_classification_blob_names(is_training=True):
    """classification blob names."""
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob:  categorical labels
        blob_names += ['labels_int32']

    return blob_names


def add_classification_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training classification models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        blobs['rois'] = im_i * blob_utils.ones((entry['gt_classes'].shape[0], 1))
        blobs['labels_int32'] = entry['gt_classes']

    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True

    return valid
