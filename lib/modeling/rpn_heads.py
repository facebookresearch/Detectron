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

from core.config import cfg
from modeling.generate_anchors import generate_anchors
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.FPN as FPN
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# RPN and Faster R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_generic_rpn_outputs(model, blob_in, dim_in, spatial_scale_in):
    """Add RPN outputs (objectness classification and bounding box regression)
    to an RPN model. Abstracts away the use of FPN.
    """
    loss_gradients = None
    if cfg.FPN.FPN_ON:
        # Delegate to the FPN module
        FPN.add_fpn_rpn_outputs(model, blob_in, dim_in, spatial_scale_in)
        if cfg.MODEL.FASTER_RCNN:
            # CollectAndDistributeFpnRpnProposals also labels proposals when in
            # training mode
            model.CollectAndDistributeFpnRpnProposals()
        if model.train:
            loss_gradients = FPN.add_fpn_rpn_losses(model)
    else:
        # Not using FPN, add RPN to a single scale
        add_single_scale_rpn_outputs(model, blob_in, dim_in, spatial_scale_in)
        if model.train:
            loss_gradients = add_single_scale_rpn_losses(model)
    return loss_gradients


def add_single_scale_rpn_outputs(model, blob_in, dim_in, spatial_scale):
    """Add RPN outputs to a single scale model (i.e., no FPN)."""
    anchors = generate_anchors(
        stride=1. / spatial_scale,
        sizes=cfg.RPN.SIZES,
        aspect_ratios=cfg.RPN.ASPECT_RATIOS
    )
    num_anchors = anchors.shape[0]
    dim_out = dim_in
    # RPN hidden representation
    model.Conv(
        blob_in,
        'conv_rpn',
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    model.Relu('conv_rpn', 'conv_rpn')
    # Proposal classification scores
    model.Conv(
        'conv_rpn',
        'rpn_cls_logits',
        dim_in,
        num_anchors,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    # Proposal bbox regression deltas
    model.Conv(
        'conv_rpn',
        'rpn_bbox_pred',
        dim_in,
        4 * num_anchors,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )

    if not model.train or cfg.MODEL.FASTER_RCNN:
        # Proposals are needed during:
        #  1) inference (== not model.train) for RPN only and Faster R-CNN
        #  OR
        #  2) training for Faster R-CNN
        # Otherwise (== training for RPN only), proposals are not needed
        model.net.Sigmoid('rpn_cls_logits', 'rpn_cls_probs')
        model.GenerateProposals(
            ['rpn_cls_probs', 'rpn_bbox_pred', 'im_info'],
            ['rpn_rois', 'rpn_roi_probs'],
            anchors=anchors,
            spatial_scale=spatial_scale
        )

    if cfg.MODEL.FASTER_RCNN:
        if model.train:
            # Add op that generates training labels for in-network RPN proposals
            model.GenerateProposalLabels(['rpn_rois', 'roidb', 'im_info'])
        else:
            # Alias rois to rpn_rois for inference
            model.net.Alias('rpn_rois', 'rois')


def add_single_scale_rpn_losses(model):
    """Add losses for a single scale RPN model (i.e., no FPN)."""
    # Spatially narrow the full-sized RPN label arrays to match the feature map
    # shape
    model.net.SpatialNarrowAs(
        ['rpn_labels_int32_wide', 'rpn_cls_logits'], 'rpn_labels_int32'
    )
    for key in ('targets', 'inside_weights', 'outside_weights'):
        model.net.SpatialNarrowAs(
            ['rpn_bbox_' + key + '_wide', 'rpn_bbox_pred'], 'rpn_bbox_' + key
        )
    loss_rpn_cls = model.net.SigmoidCrossEntropyLoss(
        ['rpn_cls_logits', 'rpn_labels_int32'],
        'loss_rpn_cls',
        scale=model.GetLossScale()
    )
    loss_rpn_bbox = model.net.SmoothL1Loss(
        [
            'rpn_bbox_pred', 'rpn_bbox_targets', 'rpn_bbox_inside_weights',
            'rpn_bbox_outside_weights'
        ],
        'loss_rpn_bbox',
        beta=1. / 9.,
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(
        model, [loss_rpn_cls, loss_rpn_bbox]
    )
    model.AddLosses(['loss_rpn_cls', 'loss_rpn_bbox'])
    return loss_gradients
