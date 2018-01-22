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

"""Test a RetinaNet network on an image database"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import json
import os
import uuid
import yaml
import logging
from collections import defaultdict, OrderedDict

from caffe2.python import core, workspace

from core.config import cfg, get_output_dir
from core.rpn_generator import _get_image_blob
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
from modeling import model_builder
from modeling.generate_anchors import generate_anchors
from pycocotools.cocoeval import COCOeval
from utils.io import save_object
from utils.timer import Timer

import utils.boxes as box_utils
import utils.c2 as c2_utils
import utils.env as envu
import utils.net as nu
import utils.subprocess as subprocess_utils

logger = logging.getLogger(__name__)


def create_cell_anchors():
    """
    Generate all types of anchors for all fpn levels/scales/aspect ratios.
    This function is called only once at the beginning of inference.
    """
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    scales_per_octave = cfg.RETINANET.SCALES_PER_OCTAVE
    aspect_ratios = cfg.RETINANET.ASPECT_RATIOS
    anchor_scale = cfg.RETINANET.ANCHOR_SCALE
    A = scales_per_octave * len(aspect_ratios)
    anchors = {}
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = np.zeros((A, 4))
        a = 0
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for aspect in aspect_ratios:
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect, )
                cell_anchors[a, :] = generate_anchors(
                    stride=stride, sizes=anchor_sizes,
                    aspect_ratios=anchor_aspect_ratios)
                a += 1
        anchors[lvl] = cell_anchors
    return anchors


def im_detections(model, im, anchors):
    """Generate RetinaNet detections on a single image."""
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    A = cfg.RETINANET.SCALES_PER_OCTAVE * len(cfg.RETINANET.ASPECT_RATIOS)
    inputs = {}
    inputs['data'], inputs['im_info'] = _get_image_blob(im)
    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))

    workspace.RunNet(model.net.Proto().name)
    scale = inputs['im_info'][0, 2]
    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)

    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)

    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        # fetch per level probability
        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        cls_prob_ravel = cls_prob.ravel()
        # In some cases [especially for very small img sizes], it's possible that
        # candidate_ind is empty if we impose threshold 0.05 at all levels. This
        # will lead to errors since no detections are found for this image. Hence,
        # for lvl 7 which has small spatial resolution, we take the threshold 0.0
        th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))
        inds = np.argpartition(
            cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
        inds = candidate_inds[inds]

        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]

        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        boxes += cell_anchors[anchor_ids, :]

        if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
            box_deltas = box_pred[0, anchor_ids, :, y, x]
        else:
            box_cls_inds = classes * 4
            box_deltas = np.vstack(
                [box_pred[0, ind:ind + 4, yi, xi]
                 for ind, yi, xi in zip(box_cls_inds, y, x)]
            )
        pred_boxes = (
            box_utils.bbox_transform(boxes, box_deltas)
            if cfg.TEST.BBOX_REG else boxes)
        pred_boxes /= scale
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

        for cls in range(1, cfg.MODEL.NUM_CLASSES):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])

    # Combine predictions across all levels and retain the top scoring by class
    detections = []
    for cls, boxes in boxes_all.items():
        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        # do class specific nms here
        keep = box_utils.nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        out = np.zeros((len(keep), 6))
        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)

    detections = np.vstack(detections)
    # sort all again
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.TEST.DETECTIONS_PER_IM], :]
    boxes = detections[:, 0:4]
    scores = detections[:, 4]
    classes = detections[:, 5]
    return boxes, scores, classes


def im_list_detections(model, im_list):
    """Generate RetinaNet proposals on all images in im_list."""
    _t = Timer()
    num_images = len(im_list)
    im_list_boxes = [[] for _ in range(num_images)]
    im_list_scores = [[] for _ in range(num_images)]
    im_list_ids = [[] for _ in range(num_images)]
    im_list_classes = [[] for _ in range(num_images)]
    # create anchors for each level
    anchors = create_cell_anchors()
    for i in range(num_images):
        im_list_ids[i] = im_list[i]['id']
        im = cv2.imread(im_list[i]['image'])
        with c2_utils.NamedCudaScope(0):
            _t.tic()
            im_list_boxes[i], im_list_scores[i], im_list_classes[i] = \
                im_detections(model, im, anchors)
            _t.toc()
        logger.info(
            'im_detections: {:d}/{:d} {:.3f}s'.format(
                i + 1, num_images, _t.average_time))
    return im_list_boxes, im_list_scores, im_list_classes, im_list_ids


def test_retinanet(ind_range=None):
    """
    Test RetinaNet model either on the entire dataset or the subset of dataset
    specified by the index range
    """
    assert cfg.RETINANET.RETINANET_ON, \
        'RETINANET_ON must be set for testing RetinaNet model'
    output_dir = get_output_dir(training=False)
    dataset = JsonDataset(cfg.TEST.DATASET)
    im_list = dataset.get_roidb()
    if ind_range is not None:
        start, end = ind_range
        im_list = im_list[start:end]
        logger.info('Testing on roidb range: {}-{}'.format(start, end))
    else:
        # if testing over the whole dataset, use the NUM_TEST_IMAGES setting
        # the NUM_TEST_IMAGES could be over a small set of images for quick
        # debugging purposes
        im_list = im_list[0:cfg.TEST.NUM_TEST_IMAGES]

    model = model_builder.create(cfg.MODEL.TYPE, train=False)
    if cfg.TEST.WEIGHTS:
        nu.initialize_from_weights_file(
            model, cfg.TEST.WEIGHTS, broadcast=False
        )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    boxes, scores, classes, image_ids = im_list_detections(
        model, im_list[0:cfg.TEST.NUM_TEST_IMAGES])

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'retinanet_detections_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'retinanet_detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(boxes=boxes, scores=scores, classes=classes, ids=image_ids, cfg=cfg_yaml),
        det_file)
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return boxes, scores, classes, image_ids


def multi_gpu_test_retinanet_on_dataset(num_images, output_dir, dataset):
    """
    If doing multi-gpu testing, we need to divide the data on various gpus and
    make the subprocess call for each child process that'll run test_retinanet()
    on its subset data. After all the subprocesses finish, we combine the results
    and return
    """
    # Retrieve the test_net binary path
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    outputs = subprocess_utils.process_in_parallel(
        'retinanet_detections', num_images, binary, output_dir)

    # Combine the results from each subprocess now
    boxes, scores, classes, image_ids = [], [], [], []
    for det_data in outputs:
        boxes.extend(det_data['boxes'])
        scores.extend(det_data['scores'])
        classes.extend(det_data['classes'])
        image_ids.extend(det_data['ids'])
    return boxes, scores, classes, image_ids,


def test_retinanet_on_dataset(multi_gpu=False):
    """
    Main entry point for testing on a given dataset: whether multi_gpu or not
    """
    output_dir = get_output_dir(training=False)
    logger.info('Output will be saved to: {:s}'.format(os.path.abspath(output_dir)))

    dataset = JsonDataset(cfg.TEST.DATASET)
    # for test-dev or full test dataset, we generate detections for all images
    if 'test-dev' in cfg.TEST.DATASET or 'test' in cfg.TEST.DATASET:
        cfg.TEST.NUM_TEST_IMAGES = len(dataset.get_roidb())

    if multi_gpu:
        num_images = cfg.TEST.NUM_TEST_IMAGES
        boxes, scores, classes, image_ids = multi_gpu_test_retinanet_on_dataset(
            num_images, output_dir, dataset
        )
    else:
        boxes, scores, classes, image_ids = test_retinanet()

    # write RetinaNet detections pkl file to be used for various purposes
    # dump the boxes first just in case there are spurious failures
    res_file = os.path.join(output_dir, 'retinanet_detections.pkl')
    logger.info(
        'Writing roidb detections to file: {}'.
        format(os.path.abspath(res_file))
    )
    save_object(
        dict(boxes=boxes, scores=scores, classes=classes, ids=image_ids),
        res_file
    )
    logger.info('Wrote RetinaNet detections to {}'.format(os.path.abspath(res_file)))

    # Write the detections to a file that can be uploaded to coco evaluation server
    # which takes a json file format
    res_file = write_coco_detection_results(
        output_dir, dataset, boxes, scores, classes, image_ids)

    # Perform coco evaluation
    coco_eval = coco_evaluate(dataset, res_file, image_ids)

    box_results = task_evaluation._coco_eval_to_box_results(coco_eval)
    return OrderedDict([(dataset.name, box_results)])


def coco_evaluate(json_dataset, res_file, image_ids):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def write_coco_detection_results(
    output_dir, json_dataset, all_boxes, all_scores, all_classes, image_ids,
    use_salt=False
):
    res_file = os.path.join(
        output_dir, 'detections_' + json_dataset.name + '_results')
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    logger.info('Writing RetinaNet detections for submitting to coco server...')
    results = []
    for (im_id, dets, cls, score) in zip(image_ids, all_boxes, all_classes, all_scores):
        dets = dets.astype(np.float)
        score = score.astype(np.float)
        classes = np.array(
            [json_dataset.contiguous_category_id_to_json_id[c] for c in cls])
        xs = dets[:, 0]
        ys = dets[:, 1]
        ws = dets[:, 2] - xs + 1
        hs = dets[:, 3] - ys + 1
        results.extend(
            [{'image_id': im_id,
              'category_id': classes[k],
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': score[k]} for k in range(dets.shape[0])])

    logger.info('Writing detection results to json: {}'.format(
        os.path.abspath(res_file)
    ))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)
    logger.info('Done!')
    return res_file
