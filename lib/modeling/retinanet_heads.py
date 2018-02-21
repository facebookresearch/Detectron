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

"""RetinaNet model heads and losses. See: https://arxiv.org/abs/1708.02002."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
import utils.blob as blob_utils


def get_retinanet_bias_init(model):
    """Initialize the biases for the conv ops that predict class probabilities.
    Initialization is performed such that at the start of training, all
    locations are predicted to be background with high probability
    (e.g., ~0.99 = 1 - cfg.RETINANET.PRIOR_PROB). See the Focal Loss paper for
    details.
    """
    prior_prob = cfg.RETINANET.PRIOR_PROB
    scales_per_octave = cfg.RETINANET.SCALES_PER_OCTAVE
    aspect_ratios = len(cfg.RETINANET.ASPECT_RATIOS)
    if cfg.RETINANET.SOFTMAX:
        # Multiclass softmax case
        bias = np.zeros((model.num_classes, 1), dtype=np.float32)
        bias[0] = np.log(
            (model.num_classes - 1) * (1 - prior_prob) / (prior_prob)
        )
        bias = np.vstack(
            [bias for _ in range(scales_per_octave * aspect_ratios)]
        )
        bias_init = (
            'GivenTensorFill', {
                'values': bias.astype(dtype=np.float32)
            }
        )
    else:
        # Per-class sigmoid (binary classification) case
        bias_init = (
            'ConstantFill', {
                'value': -np.log((1 - prior_prob) / prior_prob)
            }
        )
    return bias_init


def add_fpn_retinanet_outputs(model, blobs_in, dim_in, spatial_scales):
    """RetinaNet head. For classification and box regression, we can chose to
    have the same conv tower or a separate tower. "bl_feat_list" stores the list
    of feature blobs for bbox prediction. These blobs can be shared cls feature
    blobs if we share the tower or else are independent blobs.
    """
    dim_out = dim_in
    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
    A = len(cfg.RETINANET.ASPECT_RATIOS) * cfg.RETINANET.SCALES_PER_OCTAVE

    # compute init for bias
    bias_init = get_retinanet_bias_init(model)

    assert len(blobs_in) == k_max - k_min + 1
    bbox_feat_list = []
    cls_pred_dim = (
        model.num_classes if cfg.RETINANET.SOFTMAX else (model.num_classes - 1)
    )
    # unpacked bbox feature and add prediction layers
    bbox_regr_dim = (
        4 * (model.num_classes - 1) if cfg.RETINANET.CLASS_SPECIFIC_BBOX else 4
    )

    # ==========================================================================
    # classification tower with logits and prob prediction
    # ==========================================================================
    for lvl in range(k_min, k_max + 1):
        bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        # classification tower stack convolution starts
        for nconv in range(cfg.RETINANET.NUM_CONVS):
            suffix = 'n{}_fpn{}'.format(nconv, lvl)
            dim_in, dim_out = dim_in, dim_in
            if lvl == k_min:
                bl_out = model.Conv(
                    bl_in,
                    'retnet_cls_conv_' + suffix,
                    dim_in,
                    dim_out,
                    3,
                    stride=1,
                    pad=1,
                    weight_init=('GaussianFill', {
                        'std': 0.01
                    }),
                    bias_init=('ConstantFill', {
                        'value': 0.
                    })
                )
            else:
                bl_out = model.ConvShared(
                    bl_in,
                    'retnet_cls_conv_' + suffix,
                    dim_in,
                    dim_out,
                    3,
                    stride=1,
                    pad=1,
                    weight='retnet_cls_conv_n{}_fpn{}_w'.format(nconv, k_min),
                    bias='retnet_cls_conv_n{}_fpn{}_b'.format(nconv, k_min)
                )
            bl_in = model.Relu(bl_out, bl_out)
            bl_feat = bl_in
        # cls tower stack convolution ends. Add the logits layer now
        if lvl == k_min:
            retnet_cls_pred = model.Conv(
                bl_feat,
                'retnet_cls_pred_fpn{}'.format(lvl),
                dim_in,
                cls_pred_dim * A,
                3,
                pad=1,
                stride=1,
                weight_init=('GaussianFill', {
                    'std': 0.01
                }),
                bias_init=bias_init
            )
        else:
            retnet_cls_pred = model.ConvShared(
                bl_feat,
                'retnet_cls_pred_fpn{}'.format(lvl),
                dim_in,
                cls_pred_dim * A,
                3,
                pad=1,
                stride=1,
                weight='retnet_cls_pred_fpn{}_w'.format(k_min),
                bias='retnet_cls_pred_fpn{}_b'.format(k_min)
            )
        if not model.train:
            if cfg.RETINANET.SOFTMAX:
                model.net.GroupSpatialSoftmax(
                    retnet_cls_pred,
                    'retnet_cls_prob_fpn{}'.format(lvl),
                    num_classes=cls_pred_dim
                )
            else:
                model.net.Sigmoid(
                    retnet_cls_pred, 'retnet_cls_prob_fpn{}'.format(lvl)
                )
        if cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
            bbox_feat_list.append(bl_feat)

    # ==========================================================================
    # bbox tower if not sharing features with the classification tower with
    # logits and prob prediction
    # ==========================================================================
    if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
        for lvl in range(k_min, k_max + 1):
            bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
            for nconv in range(cfg.RETINANET.NUM_CONVS):
                suffix = 'n{}_fpn{}'.format(nconv, lvl)
                dim_in, dim_out = dim_in, dim_in
                if lvl == k_min:
                    bl_out = model.Conv(
                        bl_in,
                        'retnet_bbox_conv_' + suffix,
                        dim_in,
                        dim_out,
                        3,
                        stride=1,
                        pad=1,
                        weight_init=('GaussianFill', {
                            'std': 0.01
                        }),
                        bias_init=('ConstantFill', {
                            'value': 0.
                        })
                    )
                else:
                    bl_out = model.ConvShared(
                        bl_in,
                        'retnet_bbox_conv_' + suffix,
                        dim_in,
                        dim_out,
                        3,
                        stride=1,
                        pad=1,
                        weight='retnet_bbox_conv_n{}_fpn{}_w'.format(
                            nconv, k_min
                        ),
                        bias='retnet_bbox_conv_n{}_fpn{}_b'.format(
                            nconv, k_min
                        )
                    )
                bl_in = model.Relu(bl_out, bl_out)
                # Add octave scales and aspect ratio
                # At least 1 convolution for dealing different aspect ratios
                bl_feat = bl_in
            bbox_feat_list.append(bl_feat)
    # Depending on the features [shared/separate] for bbox, add prediction layer
    for i, lvl in enumerate(range(k_min, k_max + 1)):
        bbox_pred = 'retnet_bbox_pred_fpn{}'.format(lvl)
        bl_feat = bbox_feat_list[i]
        if lvl == k_min:
            model.Conv(
                bl_feat,
                bbox_pred,
                dim_in,
                bbox_regr_dim * A,
                3,
                pad=1,
                stride=1,
                weight_init=('GaussianFill', {
                    'std': 0.01
                }),
                bias_init=('ConstantFill', {
                    'value': 0.
                })
            )
        else:
            model.ConvShared(
                bl_feat,
                bbox_pred,
                dim_in,
                bbox_regr_dim * A,
                3,
                pad=1,
                stride=1,
                weight='retnet_bbox_pred_fpn{}_w'.format(k_min),
                bias='retnet_bbox_pred_fpn{}_b'.format(k_min)
            )


def add_fpn_retinanet_losses(model):
    loss_gradients = {}
    gradients, losses = [], []

    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid

    model.AddMetrics(['retnet_fg_num', 'retnet_bg_num'])
    # ==========================================================================
    # bbox regression loss - SelectSmoothL1Loss for multiple anchors at a location
    # ==========================================================================
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        bbox_loss = model.net.SelectSmoothL1Loss(
            [
                'retnet_bbox_pred_' + suffix,
                'retnet_roi_bbox_targets_' + suffix,
                'retnet_roi_fg_bbox_locs_' + suffix, 'retnet_fg_num'
            ],
            'retnet_loss_bbox_' + suffix,
            beta=cfg.RETINANET.BBOX_REG_BETA,
            scale=model.GetLossScale() * cfg.RETINANET.BBOX_REG_WEIGHT
        )
        gradients.append(bbox_loss)
        losses.append('retnet_loss_bbox_' + suffix)

    # ==========================================================================
    # cls loss - depends on softmax/sigmoid outputs
    # ==========================================================================
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_lvl_logits = 'retnet_cls_pred_' + suffix
        if not cfg.RETINANET.SOFTMAX:
            cls_focal_loss = model.net.SigmoidFocalLoss(
                [
                    cls_lvl_logits, 'retnet_cls_labels_' + suffix,
                    'retnet_fg_num'
                ],
                ['fl_{}'.format(suffix)],
                gamma=cfg.RETINANET.LOSS_GAMMA,
                alpha=cfg.RETINANET.LOSS_ALPHA,
                scale=model.GetLossScale(),
                num_classes=model.num_classes - 1
            )
            gradients.append(cls_focal_loss)
            losses.append('fl_{}'.format(suffix))
        else:
            cls_focal_loss, gated_prob = model.net.SoftmaxFocalLoss(
                [
                    cls_lvl_logits, 'retnet_cls_labels_' + suffix,
                    'retnet_fg_num'
                ],
                ['fl_{}'.format(suffix), 'retnet_prob_{}'.format(suffix)],
                gamma=cfg.RETINANET.LOSS_GAMMA,
                alpha=cfg.RETINANET.LOSS_ALPHA,
                scale=model.GetLossScale(),
                num_classes=model.num_classes
            )
            gradients.append(cls_focal_loss)
            losses.append('fl_{}'.format(suffix))

    loss_gradients.update(blob_utils.get_loss_gradients(model, gradients))
    model.AddLosses(losses)
    return loss_gradients
