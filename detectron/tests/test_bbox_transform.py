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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from pycocotools import mask as COCOmask

import detectron.utils.boxes as box_utils


def random_boxes(mean_box, stdev, N):
    boxes = np.random.randn(N, 4) * stdev + mean_box
    return boxes.astype(dtype=np.float32)


class TestBboxTransform(unittest.TestCase):
    def test_bbox_transform_and_inverse(self):
        weights = (5, 5, 10, 10)
        src_boxes = random_boxes([10, 10, 20, 20], 1, 10)
        dst_boxes = random_boxes([10, 10, 20, 20], 1, 10)
        deltas = box_utils.bbox_transform_inv(
            src_boxes, dst_boxes, weights=weights
        )
        dst_boxes_reconstructed = box_utils.bbox_transform(
            src_boxes, deltas, weights=weights
        )
        np.testing.assert_array_almost_equal(
            dst_boxes, dst_boxes_reconstructed, decimal=5
        )

    def test_bbox_dataset_to_prediction_roundtrip(self):
        """Simulate the process of reading a ground-truth box from a dataset,
        make predictions from proposals, convert the predictions back to the
        dataset format, and then use the COCO API to compute IoU overlap between
        the gt box and the predictions. These should have IoU of 1.
        """
        weights = (5, 5, 10, 10)
        # 1/ "read" a box from a dataset in the default (x1, y1, w, h) format
        gt_xywh_box = [10, 20, 100, 150]
        # 2/ convert it to our internal (x1, y1, x2, y2) format
        gt_xyxy_box = box_utils.xywh_to_xyxy(gt_xywh_box)
        # 3/ consider nearby proposal boxes
        prop_xyxy_boxes = random_boxes(gt_xyxy_box, 10, 10)
        # 4/ compute proposal-to-gt transformation deltas
        deltas = box_utils.bbox_transform_inv(
            prop_xyxy_boxes, np.array([gt_xyxy_box]), weights=weights
        )
        # 5/ use deltas to transform proposals to xyxy predicted box
        pred_xyxy_boxes = box_utils.bbox_transform(
            prop_xyxy_boxes, deltas, weights=weights
        )
        # 6/ convert xyxy predicted box to xywh predicted box
        pred_xywh_boxes = box_utils.xyxy_to_xywh(pred_xyxy_boxes)
        # 7/ use COCO API to compute IoU
        not_crowd = [int(False)] * pred_xywh_boxes.shape[0]
        ious = COCOmask.iou(pred_xywh_boxes, np.array([gt_xywh_box]), not_crowd)
        np.testing.assert_array_almost_equal(ious, np.ones(ious.shape))

    def test_cython_bbox_iou_against_coco_api_bbox_iou(self):
        """Check that our cython implementation of bounding box IoU overlap
        matches the COCO API implementation.
        """
        def _do_test(b1, b2):
            # Compute IoU overlap with the cython implementation
            cython_iou = box_utils.bbox_overlaps(b1, b2)
            # Compute IoU overlap with the COCO API implementation
            # (requires converting boxes from xyxy to xywh format)
            xywh_b1 = box_utils.xyxy_to_xywh(b1)
            xywh_b2 = box_utils.xyxy_to_xywh(b2)
            not_crowd = [int(False)] * b2.shape[0]
            coco_ious = COCOmask.iou(xywh_b1, xywh_b2, not_crowd)
            # IoUs should be similar
            np.testing.assert_array_almost_equal(
                cython_iou, coco_ious, decimal=5
            )

        # Test small boxes
        b1 = random_boxes([10, 10, 20, 20], 5, 10)
        b2 = random_boxes([10, 10, 20, 20], 5, 10)
        _do_test(b1, b2)

        # Test bigger boxes
        b1 = random_boxes([10, 10, 110, 20], 20, 10)
        b2 = random_boxes([10, 10, 110, 20], 20, 10)
        _do_test(b1, b2)


if __name__ == '__main__':
    unittest.main()
