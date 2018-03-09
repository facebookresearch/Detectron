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
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Functions for RPN proposal generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import core
from caffe2.python import workspace

from core.config import cfg
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from utils.blob import im_list_to_blob
from utils.io import save_object
from utils.timer import Timer
import utils.c2 as c2_utils
import utils.env as envu
import utils.net as nu
import utils.subprocess as subprocess_utils

logger = logging.getLogger(__name__)


def generate_rpn_on_dataset(output_dir, multi_gpu=False, gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(cfg.TEST.DATASET)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        _boxes, _scores, _ids, rpn_file = multi_gpu_generate_rpn_on_dataset(
            num_images, output_dir
        )
    else:
        # Processes entire dataset range by default
        _boxes, _scores, _ids, rpn_file = generate_rpn_on_range(
            output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    return evaluate_proposal_file(dataset, rpn_file, output_dir)


def multi_gpu_generate_rpn_on_dataset(num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    # Retrieve the test_net binary path
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    outputs = subprocess_utils.process_in_parallel(
        'rpn_proposals', num_images, binary, output_dir
    )

    # Collate the results from each subprocess
    boxes, scores, ids = [], [], []
    for rpn_data in outputs:
        boxes += rpn_data['boxes']
        scores += rpn_data['scores']
        ids += rpn_data['ids']
    rpn_file = os.path.join(output_dir, 'rpn_proposals.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file
    )
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def generate_rpn_on_range(output_dir, ind_range=None, gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert cfg.TEST.WEIGHTS != '', \
        'TEST.WEIGHTS must be set to the model file to test'
    assert cfg.TEST.DATASET != '', \
        'TEST.DATASET must be set to the dataset name to test'
    assert cfg.MODEL.RPN_ONLY or cfg.MODEL.FASTER_RCNN

    roidb, start_ind, end_ind, total_num_images = get_roidb(ind_range)
    logger.info(
        'Output will be saved to: {:s}'.format(os.path.abspath(output_dir))
    )

    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    nu.initialize_gpu_from_weights_file(
        model, cfg.TEST.WEIGHTS, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)

    boxes, scores, ids = generate_proposals_on_roidb(
        model,
        roidb,
        start_ind=start_ind,
        end_ind=end_ind,
        total_num_images=total_num_images,
        gpu_id=gpu_id,
    )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        rpn_name = 'rpn_proposals_range_%s_%s.pkl' % tuple(ind_range)
    else:
        rpn_name = 'rpn_proposals.pkl'
    rpn_file = os.path.join(output_dir, rpn_name)
    save_object(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file
    )
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def generate_proposals_on_roidb(
    model, roidb, start_ind=None, end_ind=None, total_num_images=None,
    gpu_id=0,
):
    """Generate RPN proposals on all images in an imdb."""
    _t = Timer()
    num_images = len(roidb)
    roidb_boxes = [[] for _ in range(num_images)]
    roidb_scores = [[] for _ in range(num_images)]
    roidb_ids = [[] for _ in range(num_images)]
    if start_ind is None:
        start_ind = 0
        end_ind = num_images
        total_num_images = num_images
    for i in range(num_images):
        roidb_ids[i] = roidb[i]['id']
        im = cv2.imread(roidb[i]['image'])
        with c2_utils.NamedCudaScope(gpu_id):
            _t.tic()
            roidb_boxes[i], roidb_scores[i] = im_proposals(model, im)
            _t.toc()
        if i % 10 == 0:
            ave_time = _t.average_time
            eta_seconds = ave_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                (
                    'rpn_generate: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, ave_time, eta
                )
            )

    return roidb_boxes, roidb_scores, roidb_ids


def im_proposals(model, im):
    """Generate RPN proposals on a single image."""
    inputs = {}
    inputs['data'], inputs['im_info'] = _get_image_blob(im)
    scale = inputs['im_info'][0, 2]
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))
    workspace.RunNet(model.net.Proto().name)

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        rois_names = [
            core.ScopedName('rpn_rois_fpn' + str(l))
            for l in range(k_min, k_max + 1)
        ]
        score_names = [
            core.ScopedName('rpn_roi_probs_fpn' + str(l))
            for l in range(k_min, k_max + 1)
        ]
        blobs = workspace.FetchBlobs(rois_names + score_names)
        # Combine predictions across all levels and retain the top scoring
        boxes = np.concatenate(blobs[:len(rois_names)])
        scores = np.concatenate(blobs[len(rois_names):]).squeeze()
        # Discussion: one could do NMS again after combining predictions from
        # the different FPN levels. Conceptually, it's probably the right thing
        # to do. For arbitrary reasons, the original FPN RPN implementation did
        # not do another round of NMS.
        inds = np.argsort(-scores)[:cfg.TEST.RPN_POST_NMS_TOP_N]
        scores = scores[inds]
        boxes = boxes[inds, :]
    else:
        boxes, scores = workspace.FetchBlobs(
            [core.ScopedName('rpn_rois'),
             core.ScopedName('rpn_roi_probs')]
        )
        scores = scores.squeeze()

    # Column 0 is the batch index in the (batch ind, x1, y1, x2, y2) encoding,
    # so we remove it since we just want to return boxes
    # Scale proposals back to the original input image scale
    boxes = boxes[:, 1:] / scale
    return boxes, scores


def get_roidb(ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(cfg.TEST.DATASET)
    roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, start, end, total_num_images


def evaluate_proposal_file(dataset, proposal_file, output_dir):
    """Evaluate box proposal average recall."""
    roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file)
    results = task_evaluation.evaluate_box_proposals(dataset, roidb)
    task_evaluation.log_box_proposal_results(results)
    recall_file = os.path.join(output_dir, 'rpn_proposal_recall.pkl')
    save_object(results, recall_file)
    return results


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[0]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info
