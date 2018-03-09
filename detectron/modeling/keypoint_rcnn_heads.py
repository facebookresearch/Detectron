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

"""Various network "heads" for predicting keypoints in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> keypoint head -> keypoint output -> loss
... -> Feature /
       Map

The keypoint head produces a feature representation of the RoI for the purpose
of keypoint prediction. The keypoint output module converts the feature
representation into keypoint heatmaps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.ResNet as ResNet
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Keypoint R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_keypoint_outputs(model, blob_in, dim):
    """Add Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    # NxKxHxW
    upsample_heatmap = (cfg.KRCNN.UP_SCALE > 1)

    if cfg.KRCNN.USE_DECONV:
        # Apply ConvTranspose to the feature representation; results in 2x
        # upsampling
        blob_in = model.ConvTranspose(
            blob_in,
            'kps_deconv',
            dim,
            cfg.KRCNN.DECONV_DIM,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        model.Relu('kps_deconv', 'kps_deconv')
        dim = cfg.KRCNN.DECONV_DIM

    if upsample_heatmap:
        blob_name = 'kps_score_lowres'
    else:
        blob_name = 'kps_score'

    if cfg.KRCNN.USE_DECONV_OUTPUT:
        # Use ConvTranspose to predict heatmaps; results in 2x upsampling
        blob_out = model.ConvTranspose(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        # Use Conv to predict heatmaps; does no upsampling
        blob_out = model.Conv(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

    if upsample_heatmap:
        # Increase heatmap output size via bilinear upsampling
        blob_out = model.BilinearInterpolation(
            blob_out, 'kps_score', cfg.KRCNN.NUM_KEYPOINTS,
            cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.UP_SCALE
        )

    return blob_out


def add_keypoint_losses(model):
    """Add Mask R-CNN keypoint specific losses."""
    # Reshape input from (N, K, H, W) to (NK, HW)
    model.net.Reshape(
        ['kps_score'], ['kps_score_reshaped', '_kps_score_old_shape'],
        shape=(-1, cfg.KRCNN.HEATMAP_SIZE * cfg.KRCNN.HEATMAP_SIZE)
    )
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    kps_prob, loss_kps = model.net.SoftmaxWithLoss(
        ['kps_score_reshaped', 'keypoint_locations_int32', 'keypoint_weights'],
        ['kps_prob', 'loss_kps'],
        scale=cfg.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS,
        spatial=0
    )
    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        model.StopGradient(
            'keypoint_loss_normalizer', 'keypoint_loss_normalizer'
        )
        loss_kps = model.net.Mul(
            ['loss_kps', 'keypoint_loss_normalizer'], 'loss_kps_normalized'
        )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_kps])
    model.AddLosses(loss_kps)
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Keypoint heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_keypoints(
    model, blob_in, dim_in, spatial_scale
):
    """Add a ResNet "conv5" / "stage5" head for Mask R-CNN keypoint prediction.
    """
    model.RoIFeatureTransform(
        blob_in,
        '_[pose]_pool5',
        blob_rois='keypoint_rois',
        method=cfg.KRCNN.ROI_XFORM_METHOD,
        resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    # Using the prefix '_[pose]_' to 'res5' enables initializing the head's
    # parameters using pretrained 'res5' parameters if given (see
    # utils.net.initialize_from_weights_file)
    s, dim_in = ResNet.add_stage(
        model,
        '_[pose]_res5',
        '_[pose]_pool5',
        3,
        dim_in,
        2048,
        512,
        cfg.KRCNN.DILATION,
        stride_init=int(cfg.KRCNN.ROI_XFORM_RESOLUTION / 7)
    )
    return s, 2048


def add_roi_pose_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """Add a Mask R-CNN keypoint head. v1convX design: X * (conv)."""
    hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
    kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[pose]_roi_feat',
        blob_rois='keypoint_rois',
        method=cfg.KRCNN.ROI_XFORM_METHOD,
        resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.KRCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim
