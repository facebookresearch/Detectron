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

"""Common utility functions for RPN and RetinaNet minibtach blobs preparation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import logging
import numpy as np
import threading

from core.config import cfg
from modeling.generate_anchors import generate_anchors
import utils.boxes as box_utils

logger = logging.getLogger(__name__)


# octave and aspect fields are only used on RetinaNet. Octave corresponds to the
# scale of the anchor and aspect denotes which aspect ratio is used in the range
# of aspect ratios
FieldOfAnchors = namedtuple(
    'FieldOfAnchors', [
        'field_of_anchors', 'num_cell_anchors', 'stride', 'field_size',
        'octave', 'aspect'
    ]
)

# Cache for memoizing _get_field_of_anchors
_threadlocal_foa = threading.local()


def get_field_of_anchors(
    stride, anchor_sizes, anchor_aspect_ratios, octave=None, aspect=None
):
    global _threadlocal_foa
    if not hasattr(_threadlocal_foa, 'cache'):
        _threadlocal_foa.cache = {}

    cache_key = str(stride) + str(anchor_sizes) + str(anchor_aspect_ratios)
    if cache_key in _threadlocal_foa.cache:
        return _threadlocal_foa.cache[cache_key]

    # Anchors at a single feature cell
    cell_anchors = generate_anchors(
        stride=stride, sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios
    )
    num_cell_anchors = cell_anchors.shape[0]

    # Generate canonical proposals from shifted anchors
    # Enumerate all shifted positions on the (H, W) grid
    fpn_max_size = cfg.FPN.COARSEST_STRIDE * np.ceil(
        cfg.TRAIN.MAX_SIZE / float(cfg.FPN.COARSEST_STRIDE)
    )
    field_size = int(np.ceil(fpn_max_size / float(stride)))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.ravel()
    shift_y = shift_y.ravel()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    # Broacast anchors over shifts to enumerate all anchors at all positions
    # in the (H, W) grid:
    #   - add A cell anchors of shape (1, A, 4) to
    #   - K shifts of shape (K, 1, 4) to get
    #   - all shifted anchors of shape (K, A, 4)
    #   - reshape to (K*A, 4) shifted anchors
    A = num_cell_anchors
    K = shifts.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    field_of_anchors = field_of_anchors.reshape((K * A, 4))
    foa = FieldOfAnchors(
        field_of_anchors=field_of_anchors.astype(np.float32),
        num_cell_anchors=num_cell_anchors,
        stride=stride,
        field_size=field_size,
        octave=octave,
        aspect=aspect
    )
    _threadlocal_foa.cache[cache_key] = foa
    return foa


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def compute_targets(ex_rois, gt_rois, weights=(1.0, 1.0, 1.0, 1.0)):
    """Compute bounding-box regression targets for an image."""
    return box_utils.bbox_transform_inv(ex_rois, gt_rois, weights).astype(
        np.float32, copy=False
    )
