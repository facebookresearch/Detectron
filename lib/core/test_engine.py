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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from core.config import cfg
from core.rpn_generator import generate_rpn_on_dataset
from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from utils.io import save_object
from utils.timer import Timer
import utils.c2 as c2_utils
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        child_func = generate_rpn_on_range
        parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def run_inference(output_dir, ind_range=None, multi_gpu_testing=False, gpu_id=0):
    parent_func, child_func = get_eval_functions()

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
            results = parent_func(output_dir, multi_gpu=multi_gpu_testing)
            all_results.update(results)

        return all_results
    else:
        # Subprocess child case:
        # In this case test_net was called via subprocess.Popen to execute on a
        # range of inputs on a single dataset (i.e., use cfg.TEST.DATASET and
        # don't loop over cfg.TEST.DATASETS)
        return child_func(output_dir, ind_range=ind_range, gpu_id=gpu_id)


def test_net_on_dataset(output_dir, multi_gpu=False, gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(cfg.TEST.DATASET)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(output_dir, gpu_id=gpu_id)
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(output_dir, ind_range=None, gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert cfg.TEST.WEIGHTS != '', \
        'TEST.WEIGHTS must be set to the model file to test'
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'
    assert cfg.TEST.DATASET != '', \
        'TEST.DATASET must be set to the dataset name to test'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        ind_range
    )
    model = initialize_model_from_cfg(gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        im = cv2.imread(entry['image'])
        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                model, im, box_proposals, timers
            )

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                segms=cls_segms_i,
                keypoints=cls_keyps_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes, all_segms, all_keyps


def initialize_model_from_cfg(gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, cfg.TEST.WEIGHTS, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    return model


def get_roidb_and_dataset(ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(cfg.TEST.DATASET)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        roidb = dataset.get_roidb(
            proposal_file=cfg.TEST.PROPOSAL_FILE,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
