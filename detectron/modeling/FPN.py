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

"""Functions for using a Feature Pyramid Network (FPN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level


# ---------------------------------------------------------------------------- #
# FPN with ResNet
# ---------------------------------------------------------------------------- #

def add_fpn_ResNet50_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet50_conv5_body, fpn_level_info_ResNet50_conv5
    )


def add_fpn_ResNet50_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5,
        P2only=True
    )


def add_fpn_ResNet101_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet101_conv5_body, fpn_level_info_ResNet101_conv5
    )


def add_fpn_ResNet101_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet101_conv5_body,
        fpn_level_info_ResNet101_conv5,
        P2only=True
    )


def add_fpn_ResNet152_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet152_conv5_body, fpn_level_info_ResNet152_conv5
    )


def add_fpn_ResNet152_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5,
        P2only=True
    )


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #

def add_fpn_onto_conv_body(
    model, conv_body_func, fpn_level_info_func, P2only=False
):
    """Add the specified conv body to the model and then add FPN levels to it.
    """
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(
        model, fpn_level_info_func()
    )

    if P2only:
        # use only the finest level
        return blobs_fpn[-1], dim_fpn, spatial_scales_fpn[-1]
    else:
        # use all levels
        return blobs_fpn, dim_fpn, spatial_scales_fpn


def add_fpn(model, fpn_level_info):
    """Add FPN connections based on the model described in the FPN paper."""
    # FPN levels are built starting from the highest/coarest level of the
    # backbone (usually "conv5"). First we build down, recursively constructing
    # lower/finer resolution FPN levels. Then we build up, constructing levels
    # that are even higher/coarser than the starting level.
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()
    # Count the number of backbone stages that we will generate FPN levels for
    # starting from the coarest backbone stage (usually the "conv5"-like level)
    # E.g., if the backbone level info defines stages 4 stages: "conv5",
    # "conv4", ... "conv2" and min_level=2, then we end up with 4 - (2 - 2) = 4
    # backbone stages to add FPN to.
    num_backbone_stages = (
        len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
    )

    lateral_input_blobs = fpn_level_info.blobs[:num_backbone_stages]
    output_blobs = [
        'fpn_inner_{}'.format(s)
        for s in fpn_level_info.blobs[:num_backbone_stages]
    ]
    fpn_dim_lateral = fpn_level_info.dims
    xavier_fill = ('XavierFill', {})

    # For the coarsest backbone level: 1x1 conv only seeds recursion
    if cfg.FPN.USE_GN:
        # use GroupNorm
        c = model.ConvGN(
            lateral_input_blobs[0],
            output_blobs[0],  # note: this is a prefix
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            group_gn=get_group_gn(fpn_dim),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        output_blobs[0] = c  # rename it
    else:
        model.Conv(
            lateral_input_blobs[0],
            output_blobs[0],
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )

    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        add_topdown_lateral_module(
            model,
            output_blobs[i],             # top-down blob
            lateral_input_blobs[i + 1],  # lateral blob
            output_blobs[i + 1],         # next output blob
            fpn_dim,                     # output dimension
            fpn_dim_lateral[i + 1]       # lateral input dimension
        )

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    spatial_scales = []
    for i in range(num_backbone_stages):
        if cfg.FPN.USE_GN:
            # use GroupNorm
            fpn_blob = model.ConvGN(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                group_gn=get_group_gn(fpn_dim),
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        else:
            fpn_blob = model.Conv(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        blobs_fpn += [fpn_blob]
        spatial_scales += [fpn_level_info.spatial_scales[i]]

    #
    # Step 2: build up starting from the coarsest backbone level
    #

    # Check if we need the P6 feature map
    if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
        # Original FPN P6 level implementation from our CVPR'17 FPN paper
        P6_blob_in = blobs_fpn[0]
        P6_name = P6_blob_in + '_subsampled_2x'
        # Use max pooling to simulate stride 2 subsampling
        P6_blob = model.MaxPool(P6_blob_in, P6_name, kernel=1, pad=0, stride=2)
        blobs_fpn.insert(0, P6_blob)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Coarser FPN levels introduced for RetinaNet
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
        fpn_blob = fpn_level_info.blobs[0]
        dim_in = fpn_level_info.dims[0]
        for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
            fpn_blob_in = fpn_blob
            if i > HIGHEST_BACKBONE_LVL + 1:
                fpn_blob_in = model.Relu(fpn_blob, fpn_blob + '_relu')
            fpn_blob = model.Conv(
                fpn_blob_in,
                'fpn_' + str(i),
                dim_in=dim_in,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=2,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, fpn_dim, spatial_scales


def add_topdown_lateral_module(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral
):
    """Add a top-down lateral module."""
    # Lateral 1x1 conv
    if cfg.FPN.USE_GN:
        # use GroupNorm
        lat = model.ConvGN(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            group_gn=get_group_gn(dim_top),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0) if cfg.FPN.ZERO_INIT_LATERAL
                else ('XavierFill', {})),
            bias_init=const_fill(0.0)
        )
    else:
        lat = model.Conv(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
        )
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn_top, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


# ---------------------------------------------------------------------------- #
# RPN with an FPN backbone
# ---------------------------------------------------------------------------- #

def add_fpn_rpn_outputs(model, blobs_in, dim_in, spatial_scales):
    """Add RPN on FPN specific outputs."""
    num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
    dim_out = dim_in

    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
    assert len(blobs_in) == k_max - k_min + 1
    for lvl in range(k_min, k_max + 1):
        bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        sc = spatial_scales[k_max - lvl]  # in reversed order
        slvl = str(lvl)

        if lvl == k_min:
            # Create conv ops with randomly initialized weights and
            # zeroed biases for the first FPN level; these will be shared by
            # all other FPN levels
            # RPN hidden representation
            conv_rpn_fpn = model.Conv(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.Conv(
                conv_rpn_fpn,
                'rpn_cls_logits_fpn' + slvl,
                dim_in,
                num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.Conv(
                conv_rpn_fpn,
                'rpn_bbox_pred_fpn' + slvl,
                dim_in,
                4 * num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
        else:
            # Share weights and biases
            sk_min = str(k_min)
            # RPN hidden representation
            conv_rpn_fpn = model.ConvShared(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight='conv_rpn_fpn' + sk_min + '_w',
                bias='conv_rpn_fpn' + sk_min + '_b'
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.ConvShared(
                conv_rpn_fpn,
                'rpn_cls_logits_fpn' + slvl,
                dim_in,
                num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight='rpn_cls_logits_fpn' + sk_min + '_w',
                bias='rpn_cls_logits_fpn' + sk_min + '_b'
            )
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.ConvShared(
                conv_rpn_fpn,
                'rpn_bbox_pred_fpn' + slvl,
                dim_in,
                4 * num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight='rpn_bbox_pred_fpn' + sk_min + '_w',
                bias='rpn_bbox_pred_fpn' + sk_min + '_b'
            )

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Proposals are needed during:
            #  1) inference (== not model.train) for RPN only and Faster R-CNN
            #  OR
            #  2) training for Faster R-CNN
            # Otherwise (== training for RPN only), proposals are not needed
            lvl_anchors = generate_anchors(
                stride=2.**lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            rpn_cls_probs_fpn = model.net.Sigmoid(
                rpn_cls_logits_fpn, 'rpn_cls_probs_fpn' + slvl
            )
            model.GenerateProposals(
                [rpn_cls_probs_fpn, rpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc
            )


def add_fpn_rpn_losses(model):
    """Add RPN on FPN specific losses."""
    loss_gradients = {}
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map
        # shape
        model.net.SpatialNarrowAs(
            ['rpn_labels_int32_wide_fpn' + slvl, 'rpn_cls_logits_fpn' + slvl],
            'rpn_labels_int32_fpn' + slvl
        )
        for key in ('targets', 'inside_weights', 'outside_weights'):
            model.net.SpatialNarrowAs(
                [
                    'rpn_bbox_' + key + '_wide_fpn' + slvl,
                    'rpn_bbox_pred_fpn' + slvl
                ],
                'rpn_bbox_' + key + '_fpn' + slvl
            )
        loss_rpn_cls_fpn = model.net.SigmoidCrossEntropyLoss(
            ['rpn_cls_logits_fpn' + slvl, 'rpn_labels_int32_fpn' + slvl],
            'loss_rpn_cls_fpn' + slvl,
            normalize=0,
            scale=(
                model.GetLossScale() / cfg.TRAIN.RPN_BATCH_SIZE_PER_IM /
                cfg.TRAIN.IMS_PER_BATCH
            )
        )
        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_rpn_bbox_fpn = model.net.SmoothL1Loss(
            [
                'rpn_bbox_pred_fpn' + slvl, 'rpn_bbox_targets_fpn' + slvl,
                'rpn_bbox_inside_weights_fpn' + slvl,
                'rpn_bbox_outside_weights_fpn' + slvl
            ],
            'loss_rpn_bbox_fpn' + slvl,
            beta=1. / 9.,
            scale=model.GetLossScale(),
        )
        loss_gradients.update(
            blob_utils.
            get_loss_gradients(model, [loss_rpn_cls_fpn, loss_rpn_bbox_fpn])
        )
        model.AddLosses(['loss_rpn_cls_fpn' + slvl, 'loss_rpn_bbox_fpn' + slvl])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Helper functions for working with multilevel FPN RoIs
# ---------------------------------------------------------------------------- #

def map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    s = np.sqrt(box_utils.boxes_area(rois))
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls


def add_multilevel_roi_blobs(
    blobs, blob_prefix, rois, target_lvls, lvl_min, lvl_max
):
    """Add RoI blobs for multiple FPN levels to the blobs dict.

    blobs: a dict mapping from blob name to numpy ndarray
    blob_prefix: name prefix to use for the FPN blobs
    rois: the source rois as a 2D numpy array of shape (N, 5) where each row is
      an roi and the columns encode (batch_idx, x1, y1, x2, y2)
    target_lvls: numpy array of shape (N, ) indicating which FPN level each roi
      in rois should be assigned to
    lvl_min: the finest (highest resolution) FPN level (e.g., 2)
    lvl_max: the coarest (lowest resolution) FPN level (e.g., 6)
    """
    rois_idx_order = np.empty((0, ))
    rois_stacked = np.zeros((0, 5), dtype=np.float32)  # for assert
    for lvl in range(lvl_min, lvl_max + 1):
        idx_lvl = np.where(target_lvls == lvl)[0]
        blobs[blob_prefix + '_fpn' + str(lvl)] = rois[idx_lvl, :]
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_stacked = np.vstack(
            [rois_stacked, blobs[blob_prefix + '_fpn' + str(lvl)]]
        )
    rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
    blobs[blob_prefix + '_idx_restore_int32'] = rois_idx_restore
    # Sanity check that restore order is correct
    assert (rois_stacked[rois_idx_restore] == rois).all()


# ---------------------------------------------------------------------------- #
# FPN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
