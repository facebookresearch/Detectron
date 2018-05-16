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

"""Compute minibatch blobs for training a RetinaNet network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

import detectron.utils.boxes as box_utils
import detectron.roi_data.data_utils as data_utils
from detectron.core.config import cfg


logger = logging.getLogger(__name__)


def get_retinanet_blob_names(is_training=True):
    """
    Returns blob names in the order in which they are read by the data
    loader.

    N = number of images per minibatch
    A = number of anchors = num_scales * num_aspect_ratios
        (for example 9 used in RetinaNet paper)
    H, W = spatial dimensions (different for each FPN level)
    M = Out of all the anchors generated, depending on the positive/negative IoU
        overlap thresholds, we will have M positive anchors. These are the anchors
        that bounding box branch will regress on.

    retnet_cls_labels -> labels for the cls branch for each FPN level
                         Shape: N x A x H x W

    retnet_roi_bbox_targets -> targets for the bbox regression branch
                               Shape: M x 4

    retnet_roi_fg_bbox_locs -> for the bbox regression, since we are only
                               interested in regressing on fg bboxes which are
                               M in number and the output prediction of the network
                               is of shape N x (A * 4) x H x W
                               (in case of non class-specific bbox), so we
                               store the locations of positive fg boxes in this
                               blob retnet_roi_fg_bbox_locs of shape M x 4 where
                               each row looks like: [img_id, anchor_id, x_loc, y_loc]
    """
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    assert cfg.FPN.FPN_ON, "RetinaNet uses FPN for dense detection"
    # Same format as RPN blobs, but one per FPN level
    if is_training:
        blob_names += ['retnet_fg_num', 'retnet_bg_num']
        for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
            suffix = 'fpn{}'.format(lvl)
            blob_names += [
                'retnet_cls_labels_' + suffix,
                'retnet_roi_bbox_targets_' + suffix,
                'retnet_roi_fg_bbox_locs_' + suffix,
            ]
    return blob_names


def add_retinanet_blobs(blobs, im_scales, roidb, image_width, image_height):
    """Add RetinaNet blobs."""
    # RetinaNet is applied to many feature levels, as in the FPN paper
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    scales_per_octave = cfg.RETINANET.SCALES_PER_OCTAVE
    num_aspect_ratios = len(cfg.RETINANET.ASPECT_RATIOS)
    aspect_ratios = cfg.RETINANET.ASPECT_RATIOS
    anchor_scale = cfg.RETINANET.ANCHOR_SCALE

    # get anchors from all levels for all scales/aspect ratios
    foas = []
    for lvl in range(k_min, k_max + 1):
        stride = 2. ** lvl
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for idx in range(num_aspect_ratios):
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect_ratios[idx], )
                foa = data_utils.get_field_of_anchors(
                    stride, anchor_sizes, anchor_aspect_ratios, octave, idx)
                foas.append(foa)
    all_anchors = np.concatenate([f.field_of_anchors for f in foas])

    blobs['retnet_fg_num'], blobs['retnet_bg_num'] = 0.0, 0.0
    for im_i, entry in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        assert len(gt_inds) > 0, \
            'Empty ground truth empty for image is not allowed. Please check.'

        gt_rois = entry['boxes'][gt_inds, :] * scale
        gt_classes = entry['gt_classes'][gt_inds]

        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        retinanet_blobs, fg_num, bg_num = _get_retinanet_blobs(
            foas, all_anchors, gt_rois, gt_classes, image_width, image_height)
        for i, foa in enumerate(foas):
            for k, v in retinanet_blobs[i].items():
                # the way it stacks is:
                # [[anchors for image1] + [anchors for images 2]]
                level = int(np.log2(foa.stride))
                key = '{}_fpn{}'.format(k, level)
                if k == 'retnet_roi_fg_bbox_locs':
                    v[:, 0] = im_i
                    # loc_stride: 80 * 4 if cls_specific else 4
                    loc_stride = 4  # 4 coordinate corresponding to bbox prediction
                    if cfg.RETINANET.CLASS_SPECIFIC_BBOX:
                        loc_stride *= (cfg.MODEL.NUM_CLASSES - 1)
                    anchor_ind = foa.octave * num_aspect_ratios + foa.aspect
                    # v[:, 1] is the class label [range 0-80] if we do
                    # class-specfic bbox otherwise it is 0. In case of class
                    # specific, based on the label, the location of current
                    # anchor is class_label * 4 and then we take into account
                    # the anchor_ind if the anchors
                    v[:, 1] *= 4
                    v[:, 1] += loc_stride * anchor_ind
                blobs[key].append(v)
        blobs['retnet_fg_num'] += fg_num
        blobs['retnet_bg_num'] += bg_num

    blobs['retnet_fg_num'] = blobs['retnet_fg_num'].astype(np.float32)
    blobs['retnet_bg_num'] = blobs['retnet_bg_num'].astype(np.float32)

    N = len(roidb)
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            # compute number of anchors
            A = int(len(v) / N)
            # for the cls branch labels [per fpn level],
            # we have blobs['retnet_cls_labels_fpn{}'] as a list until this step
            # and length of this list is N x A where
            # N = num_images, A = num_anchors for example, N = 2, A = 9
            # Each element of the list has the shape 1 x 1 x H x W where H, W are
            # spatial dimension of curret fpn lvl. Let a{i} denote the element
            # corresponding to anchor i [9 anchors total] in the list.
            # The elements in the list are in order [[a0, ..., a9], [a0, ..., a9]]
            # however the network will make predictions like 2 x (9 * 80) x H x W
            # so we first concatenate the elements of each image to a numpy array
            # and then concatenate the two images to get the 2 x 9 x H x W

            if k.find('retnet_cls_labels') >= 0:
                tmp = []
                # concat anchors within an image
                for i in range(0, len(v), A):
                    tmp.append(np.concatenate(v[i: i + A], axis=1))
                # concat images
                blobs[k] = np.concatenate(tmp, axis=0)
            else:
                # for the bbox branch elements [per FPN level],
                #  we have the targets and the fg boxes locations
                # in the shape: M x 4 where M is the number of fg locations in a
                # given image at the current FPN level. For the given level,
                # the bbox predictions will be. The elements in the list are in
                # order [[a0, ..., a9], [a0, ..., a9]]
                # Concatenate them to form M x 4
                blobs[k] = np.concatenate(v, axis=0)
    return True


def _get_retinanet_blobs(
        foas, all_anchors, gt_boxes, gt_classes, im_width, im_height):
    total_anchors = all_anchors.shape[0]
    logger.debug('Getting mad blobs: im_height {} im_width: {}'.format(
        im_height, im_width))

    inds_inside = np.arange(all_anchors.shape[0])
    anchors = all_anchors
    num_inside = len(inds_inside)

    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.float32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[
            np.arange(num_inside), anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max)[0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        gt_inds = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds]
        # Fg label: above threshold IOU
        inds = anchor_to_gt_max >= cfg.RETINANET.POSITIVE_OVERLAP
        gt_inds = anchor_to_gt_argmax[inds]
        labels[inds] = gt_classes[gt_inds]

    fg_inds = np.where(labels >= 1)[0]
    bg_inds = np.where(anchor_to_gt_max < cfg.RETINANET.NEGATIVE_OVERLAP)[0]
    labels[bg_inds] = 0
    num_fg, num_bg = len(fg_inds), len(bg_inds)

    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_targets[fg_inds, :] = data_utils.compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])

    # Map up to original set of anchors
    labels = data_utils.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = data_utils.unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        end_idx = start_idx + H * W
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        start_idx = end_idx

        # labels output with shape (1, height, width)
        _labels = _labels.reshape((1, 1, H, W))
        # bbox_targets output with shape (1, 4 * A, height, width)
        _bbox_targets = _bbox_targets.reshape((1, H, W, 4)).transpose(0, 3, 1, 2)
        stride = foa.stride
        w = int(im_width / stride)
        h = int(im_height / stride)

        # data for select_smooth_l1 loss
        num_classes = cfg.MODEL.NUM_CLASSES - 1
        inds_4d = np.where(_labels > 0)
        M = len(inds_4d)
        _roi_bbox_targets = np.zeros((0, 4))
        _roi_fg_bbox_locs = np.zeros((0, 4))
        if M > 0:
            im_inds, y, x = inds_4d[0], inds_4d[2], inds_4d[3]
            _roi_bbox_targets = np.zeros((len(im_inds), 4))
            _roi_fg_bbox_locs = np.zeros((len(im_inds), 4))
            lbls = _labels[im_inds, :, y, x]
            for i, lbl in enumerate(lbls):
                l = lbl[0] - 1
                if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
                    l = 0
                assert l >= 0 and l < num_classes, 'label out of the range'
                _roi_bbox_targets[i, :] = _bbox_targets[:, :, y[i], x[i]]
                _roi_fg_bbox_locs[i, :] = np.array([[0, l, y[i], x[i]]])
        blobs_out.append(
            dict(
                retnet_cls_labels=_labels[:, :, 0:h, 0:w].astype(np.int32),
                retnet_roi_bbox_targets=_roi_bbox_targets.astype(np.float32),
                retnet_roi_fg_bbox_locs=_roi_fg_bbox_locs.astype(np.float32),
            ))
    out_num_fg = np.array([num_fg + 1.0], dtype=np.float32)
    out_num_bg = (
        np.array([num_bg + 1.0]) * (cfg.MODEL.NUM_CLASSES - 1) +
        out_num_fg * (cfg.MODEL.NUM_CLASSES - 2))
    return blobs_out, out_num_fg, out_num_bg
