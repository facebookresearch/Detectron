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
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from past.builtins import basestring
from utils.collections import AttrDict
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

from utils.io import cache_url

logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this .pkl file
__C.TRAIN.WEIGHTS = b''

# Datasets to train on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images *per GPU* in the training minibatch
# Total images per minibatch = TRAIN.IMS_PER_BATCH * NUM_GPUS
__C.TRAIN.IMS_PER_BATCH = 2

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
__C.TRAIN.BATCH_SIZE_PER_IM = 64

# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for an RoI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for an RoI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Overlap required between an RoI and a ground-truth box in order for that
# (RoI, gt box) pair to be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Snapshot (model checkpoint) period
# Divide by NUM_GPUS to determine actual period (e.g., 20000/8 => 2500 iters)
# to allow for linear training schedule scaling
__C.TRAIN.SNAPSHOT_ITERS = 20000

# Train using these proposals
# During training, all proposals specified in the file are used (no limit is
# applied)
# Proposal files must be in correspondence with the datasets listed in
# TRAIN.DATASETS
__C.TRAIN.PROPOSAL_FILES = ()

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide)
# This feature is critical for saving memory (and makes training slightly
# faster)
__C.TRAIN.ASPECT_GROUPING = True

# ---------------------------------------------------------------------------- #
# RPN training options
# ---------------------------------------------------------------------------- #

# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IOU >= thresh ==> positive RPN
# example)
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IOU < thresh ==> negative RPN
# example)
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Target fraction of foreground (positive) examples per RPN minibatch
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of RPN examples per image
__C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256

# NMS threshold used on RPN proposals (used during end-to-end training with RPN)
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring RPN proposals to keep after applying NMS
# This is the total number of RPN proposals produced (for both FPN and non-FPN
# cases)
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.TRAIN.RPN_STRADDLE_THRESH = 0

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale; not scale used during training or inference)
__C.TRAIN.RPN_MIN_SIZE = 0

# Filter proposals that are inside of crowd regions by CROWD_FILTER_THRESH
# "Inside" is measured as: proposal-with-crowd intersection area divided by
# proposal area
__C.TRAIN.CROWD_FILTER_THRESH = 0.7

# Ignore ground-truth objects with area < this threshold
__C.TRAIN.GT_MIN_AREA = -1

# Freeze the backbone architecture during training if set to True
__C.TRAIN.FREEZE_CONV_BODY = False

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
__C.TRAIN.AUTO_RESUME = True


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 4


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this .pkl file
__C.TEST.WEIGHTS = b''

# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()

# Scales to use during testing
# Each scale is the pixel size of an image's shortest side
# If multiple scales are given, then all scales are used as in multiscale
# inference
__C.TEST.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Apply Fast R-CNN style bounding-box regression if True
__C.TEST.BBOX_REG = True

# Test using these proposal files (must correspond with TEST.DATASETS)
__C.TEST.PROPOSAL_FILES = ()

# Limit on the number of proposals per image used during inference
__C.TEST.PROPOSAL_LIMIT = 2000

# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.TEST.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring RPN proposals to keep after applying NMS
# This is the total number of RPN proposals produced (for both FPN and non-FPN
# cases)
__C.TEST.RPN_POST_NMS_TOP_N = 2000

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale; not scale used during training or inference)
__C.TEST.RPN_MIN_SIZE = 0

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.TEST.DETECTIONS_PER_IM = 100

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.TEST.SCORE_THRESH = 0.05

# Save detection results files if True
# If false, results files are cleaned up (they can be large) after local
# evaluation
__C.TEST.COMPETITION_MODE = True

# Evaluate detections with the COCO json dataset eval code even if it's not the
# evaluation code for the dataset (e.g. evaluate PASCAL VOC results using the
# COCO API to get COCO style AP on PASCAL VOC)
__C.TEST.FORCE_JSON_DATASET_EVAL = False

# [Inferred value; do not set directly in a config]
# Indicates if precomputed proposals are used at test time
# Not set for 1-stage models and 2-stage models with RPN subnetwork enabled
__C.TEST.PRECOMPUTED_PROPOSALS = True

# [Inferred value; do not set directly in a config]
# Active dataset to test on
__C.TEST.DATASET = b''

# [Inferred value; do not set directly in a config]
# Active proposal file to use
__C.TEST.PROPOSAL_FILE = b''


# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_2x.yaml for an example
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_AUG = AttrDict()

# Enable test-time augmentation for bounding box detection if True
__C.TEST.BBOX_AUG.ENABLED = False

# Heuristic used to combine predicted box scores
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.SCORE_HEUR = b'UNION'

# Heuristic used to combine predicted box coordinates
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.COORD_HEUR = b'UNION'

# Horizontal flip at the original scale (id transform)
__C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
__C.TEST.BBOX_AUG.SCALE_H_FLIP = False

# Apply scaling based on object size
__C.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
__C.TEST.BBOX_AUG.AREA_TH_LO = 50**2
__C.TEST.BBOX_AUG.AREA_TH_HI = 180**2

# Each aspect ratio is relative to image width
__C.TEST.BBOX_AUG.ASPECT_RATIOS = ()

# Horizontal flip at each aspect ratio
__C.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False

# ---------------------------------------------------------------------------- #
# Test-time augmentations for mask detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_2x.yaml for an example
# ---------------------------------------------------------------------------- #
__C.TEST.MASK_AUG = AttrDict()

# Enable test-time augmentation for instance mask detection if True
__C.TEST.MASK_AUG.ENABLED = False

# Heuristic used to combine mask predictions
# SOFT prefix indicates that the computation is performed on soft masks
#   Valid options: ('SOFT_AVG', 'SOFT_MAX', 'LOGIT_AVG')
__C.TEST.MASK_AUG.HEUR = b'SOFT_AVG'

# Horizontal flip at the original scale (id transform)
__C.TEST.MASK_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.MASK_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.MASK_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
__C.TEST.MASK_AUG.SCALE_H_FLIP = False

# Apply scaling based on object size
__C.TEST.MASK_AUG.SCALE_SIZE_DEP = False
__C.TEST.MASK_AUG.AREA_TH = 180**2

# Each aspect ratio is relative to image width
__C.TEST.MASK_AUG.ASPECT_RATIOS = ()

# Horizontal flip at each aspect ratio
__C.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP = False

# ---------------------------------------------------------------------------- #
# Test-augmentations for keypoints detection
# configs/test_time_aug/keypoint_rcnn_R-50-FPN_1x.yaml
# ---------------------------------------------------------------------------- #
__C.TEST.KPS_AUG = AttrDict()

# Enable test-time augmentation for keypoint detection if True
__C.TEST.KPS_AUG.ENABLED = False

# Heuristic used to combine keypoint predictions
#   Valid options: ('HM_AVG', 'HM_MAX')
__C.TEST.KPS_AUG.HEUR = b'HM_AVG'

# Horizontal flip at the original scale (id transform)
__C.TEST.KPS_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.KPS_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.KPS_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
__C.TEST.KPS_AUG.SCALE_H_FLIP = False

# Apply scaling based on object size
__C.TEST.KPS_AUG.SCALE_SIZE_DEP = False
__C.TEST.KPS_AUG.AREA_TH = 180**2

# Eeach aspect ratio is realtive to image width
__C.TEST.KPS_AUG.ASPECT_RATIOS = ()

# Horizontal flip at each aspect ratio
__C.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP = False

# ---------------------------------------------------------------------------- #
# Soft NMS
# ---------------------------------------------------------------------------- #
__C.TEST.SOFT_NMS = AttrDict()

# Use soft NMS instead of standard NMS if set to True
__C.TEST.SOFT_NMS.ENABLED = False
# See soft NMS paper for definition of these options
__C.TEST.SOFT_NMS.METHOD = b'linear'
__C.TEST.SOFT_NMS.SIGMA = 0.5
# For the soft NMS overlap threshold, we simply use TEST.NMS

# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_VOTE = AttrDict()

# Use box voting if set to True
__C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
__C.TEST.BBOX_VOTE.SCORING_METHOD = b'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_rcnn', 'mask_rcnn', ...)
__C.MODEL.TYPE = b''

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
__C.MODEL.CONV_BODY = b''

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# Use a class agnostic bounding box regressor instead of the default per-class
# regressor
__C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.MODEL.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# The meaning of FASTER_RCNN depends on the context (training vs. inference):
# 1) During training, FASTER_RCNN = True means that end-to-end training will be
#    used to jointly train the RPN subnetwork and the Fast R-CNN subnetwork
#    (Faster R-CNN = RPN + Fast R-CNN).
# 2) During inference, FASTER_RCNN = True means that the model's RPN subnetwork
#    will be used to generate proposals rather than relying on precomputed
#    proposals. Note that FASTER_RCNN = True can be used at inference time even
#    if the Faster R-CNN model was trained with stagewise training (which
#    consists of alternating between RPN and Fast R-CNN training in a way that
#    finally leads to a single network).
__C.MODEL.FASTER_RCNN = False

# Indicates the model makes instance mask predictions (as in Mask R-CNN)
__C.MODEL.MASK_ON = False

# Indicates the model makes keypoint predictions (as in Mask R-CNN for
# keypoints)
__C.MODEL.KEYPOINTS_ON = False

# Indicates the model's computation terminates with the production of RPN
# proposals (i.e., it outputs proposals ONLY, no actual object detections)
__C.MODEL.RPN_ONLY = False

# Caffe2 net execution type
# Use 'prof_dag' to get profiling statistics
__C.MODEL.EXECUTION_TYPE = b'dag'


# ---------------------------------------------------------------------------- #
# RetinaNet options
# ---------------------------------------------------------------------------- #
__C.RETINANET = AttrDict()

# RetinaNet is used (instead of Fast/er/Mask R-CNN/R-FCN/RPN) if True
__C.RETINANET.RETINANET_ON = False

# Anchor aspect ratios to use
__C.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)

# Anchor scales per octave
__C.RETINANET.SCALES_PER_OCTAVE = 3

# At each FPN level, we generate anchors based on their scale, aspect_ratio,
# stride of the level, and we multiply the resulting anchor by ANCHOR_SCALE
__C.RETINANET.ANCHOR_SCALE = 4

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
__C.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
__C.RETINANET.BBOX_REG_WEIGHT = 1.0

# Smooth L1 loss beta for bbox regression
__C.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
__C.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
__C.RETINANET.POSITIVE_OVERLAP = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
__C.RETINANET.NEGATIVE_OVERLAP = 0.4

# Focal loss parameter: alpha
__C.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
__C.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
__C.RETINANET.PRIOR_PROB = 0.01

# Whether classification and bbox branch tower should be shared or not
__C.RETINANET.SHARE_CLS_BBOX_TOWER = False

# Use class specific bounding box regression instead of the default class
# agnostic regression
__C.RETINANET.CLASS_SPECIFIC_BBOX = False

# Whether softmax should be used in classification branch training
__C.RETINANET.SOFTMAX = False

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
__C.RETINANET.INFERENCE_TH = 0.05


# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = b'step'

# Some LR Policies (by example):
# 'step'
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** (cur_iter // SOLVER.STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies
__C.SOLVER.STEPS = []

# Learning rates to use with 'steps_with_lrs' policy
__C.SOLVER.LRS = []

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constant' or 'linear' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'linear'

# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.SOLVER.SCALE_MOMENTUM = True
# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large)
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()

# The type of RoI head to use for bounding box classification and regression
# The string must match a function this is imported in modeling.model_builder
# (e.g., 'head_builder.add_roi_2mlp_head' to specify a two hidden layer MLP)
__C.FAST_RCNN.ROI_BOX_HEAD = b''

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.MLP_HEAD_DIM = 1024

# RoI transformation function (e.g., RoIPool or RoIAlign)
# (RoIPoolF is the same as RoIPool; ignore the trailing 'F')
__C.FAST_RCNN.ROI_XFORM_METHOD = b'RoIPoolF'

# Number of grid sampling points in RoIAlign (usually use 2)
# Only applies to RoIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transform output resolution
# Note: some models may have constraints on what they can use, e.g. they use
# pretrained FC layers like in VGG16, and will ignore this option
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()

# [Infered value; do not set directly in a config]
# Indicates that the model contains an RPN subnetwork
__C.RPN.RPN_ON = False

# RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
# Note: these options are *not* used by FPN RPN; see FPN.RPN* options
__C.RPN.SIZES = (64, 128, 256, 512)

# Stride of the feature map that RPN is attached
__C.RPN.STRIDE = 16

# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1, 2)


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()

# FPN is enabled if True
__C.FPN.FPN_ON = False

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# Initialize the lateral connections to output zero if True
__C.FPN.ZERO_INIT_LATERAL = False

# Stride of the coarsest FPN level
# This is needed so the input can be padded properly
__C.FPN.COARSEST_STRIDE = 32

#
# FPN may be used for just RPN, just object detection, or both
#

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = False
# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0
__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5
# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = False
# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6
# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2
# FPN RPN anchor aspect ratios
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this size on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
__C.FPN.RPN_ANCHOR_START_SIZE = 32
# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False


# ---------------------------------------------------------------------------- #
# Mask R-CNN options ("MRCNN" means Mask R-CNN)
# ---------------------------------------------------------------------------- #
__C.MRCNN = AttrDict()

# The type of RoI head to use for instance mask prediction
# The string must match a function this is imported in modeling.model_builder
# (e.g., 'mask_rcnn_heads.ResNet_mask_rcnn_fcn_head_v1up4convs')
__C.MRCNN.ROI_MASK_HEAD = b''

# Resolution of mask predictions
__C.MRCNN.RESOLUTION = 14

# RoI transformation function and associated options
__C.MRCNN.ROI_XFORM_METHOD = b'RoIAlign'

# RoI transformation function (e.g., RoIPool or RoIAlign)
__C.MRCNN.ROI_XFORM_RESOLUTION = 7

# Number of grid sampling points in RoIAlign (usually use 2)
# Only applies to RoIAlign
__C.MRCNN.ROI_XFORM_SAMPLING_RATIO = 0

# Number of channels in the mask head
__C.MRCNN.DIM_REDUCED = 256

# Use dilated convolution in the mask head
__C.MRCNN.DILATION = 2

# Upsample the predicted masks by this factor
__C.MRCNN.UPSAMPLE_RATIO = 1

# Use a fully-connected layer to predict the final masks instead of a conv layer
__C.MRCNN.USE_FC_OUTPUT = False

# Weight initialization method for the mask head and mask output layers
__C.MRCNN.CONV_INIT = b'GaussianFill'

# Use class specific mask predictions if True (otherwise use class agnostic mask
# predictions)
__C.MRCNN.CLS_SPECIFIC_MASK = True

# Multi-task loss weight for masks
__C.MRCNN.WEIGHT_LOSS_MASK = 1.0

# Binarization threshold for converting soft masks to hard masks
__C.MRCNN.THRESH_BINARIZE = 0.5


# ---------------------------------------------------------------------------- #
# Keyoint Mask R-CNN options ("KRCNN" = Mask R-CNN with Keypoint support)
# ---------------------------------------------------------------------------- #
__C.KRCNN = AttrDict()

# The type of RoI head to use for instance keypoint prediction
# The string must match a function this is imported in modeling.model_builder
# (e.g., 'keypoint_rcnn_heads.add_roi_pose_head_v1convX')
__C.KRCNN.ROI_KEYPOINTS_HEAD = b''

# Output size (and size loss is computed on), e.g., 56x56
__C.KRCNN.HEATMAP_SIZE = -1

# Use bilinear interpolation to upsample the final heatmap by this factor
__C.KRCNN.UP_SCALE = -1

# Apply a ConvTranspose layer to the hidden representation computed by the
# keypoint head prior to predicting the per-keypoint heatmaps
__C.KRCNN.USE_DECONV = False
# Channel dimension of the hidden representation produced by the ConvTranspose
__C.KRCNN.DECONV_DIM = 256

# Use a ConvTranspose layer to predict the per-keypoint heatmaps
__C.KRCNN.USE_DECONV_OUTPUT = False

# Use dilation in the keypoint head
__C.KRCNN.DILATION = 1

# Size of the kernels to use in all ConvTranspose operations
__C.KRCNN.DECONV_KERNEL = 4

# Number of keypoints in the dataset (e.g., 17 for COCO)
__C.KRCNN.NUM_KEYPOINTS = -1

# Number of stacked Conv layers in keypoint head
__C.KRCNN.NUM_STACKED_CONVS = 8

# Dimension of the hidden representation output by the keypoint head
__C.KRCNN.CONV_HEAD_DIM = 256

# Conv kernel size used in the keypoint head
__C.KRCNN.CONV_HEAD_KERNEL = 3
# Conv kernel weight filling function
__C.KRCNN.CONV_INIT = b'GaussianFill'

# Use NMS based on OKS if True
__C.KRCNN.NMS_OKS = False

# Source of keypoint confidence
#   Valid options: ('bbox', 'logit', 'prob')
__C.KRCNN.KEYPOINT_CONFIDENCE = b'bbox'

# Standard ROI XFORM options (see FAST_RCNN or MRCNN options)
__C.KRCNN.ROI_XFORM_METHOD = b'RoIAlign'
__C.KRCNN.ROI_XFORM_RESOLUTION = 7
__C.KRCNN.ROI_XFORM_SAMPLING_RATIO = 0

# Minimum number of labeled keypoints that must exist in a minibatch (otherwise
# the minibatch is discarded)
__C.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH = 20

# When infering the keypoint locations from the heatmap, don't scale the heatmap
# below this minimum size
__C.KRCNN.INFERENCE_MIN_SIZE = 0

# Multi-task loss weight to use for keypoints
# Recommended values:
#   - use 1.0 if KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is True
#   - use 4.0 if KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False
__C.KRCNN.LOSS_WEIGHT = 1.0

# Normalize by the total number of visible keypoints in the minibatch if True.
# Otherwise, normalize by the total number of keypoints that could ever exist
# in the minibatch. See comments in modeling.model_builder.add_keypoint_losses
# for detailed discussion.
__C.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS = True


# ---------------------------------------------------------------------------- #
# R-FCN options
# ---------------------------------------------------------------------------- #
__C.RFCN = AttrDict()

# Position-sensitive RoI pooling output grid size (height and width)
__C.RFCN.PS_GRID_SIZE = 3


# ---------------------------------------------------------------------------- #
# ResNets options ("ResNets" = ResNet and ResNeXt)
# ---------------------------------------------------------------------------- #
__C.RESNETS = AttrDict()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
__C.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
__C.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
__C.RESNETS.STRIDE_1X1 = True

# Residual transformation function
__C.RESNETS.TRANS_FUNC = b'bottleneck_transformation'

# Apply dilation in stage "res5"
__C.RESNETS.RES5_DILATION = 1


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Use NCCL for all reduce, otherwise use muji
# Warning: if set to True, you may experience deadlocks
__C.USE_NCCL = False

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1 / 16.

# Clip bounding box transformation predictions to prevent np.exp from
# overflowing
# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# "Fun" fact: the history of where these values comes from is lost
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Output basedir
__C.OUTPUT_DIR = b'/tmp'

# Name (or path to) the matlab executable
__C.MATLAB = b'matlab'

# Reduce memory usage with memonger gradient blob sharing
__C.MEMONGER = True

# Futher reduce memory by allowing forward pass activations to be shared when
# possible. Note that this will cause activation blob inspection (values,
# shapes, etc.) to be meaningless when activation blobs are reused.
__C.MEMONGER_SHARE_ACTIVATIONS = False

# Dump detection visualizations
__C.VIS = False

# Score threshold for visualization
__C.VIS_TH = 0.9

# Expected results should take the form of a list of expectations, each
# specified by four elements (dataset, task, metric, expected value). For
# example: [['coco_2014_minival', 'box_proposal', 'AR@1000', 0.387]]
__C.EXPECTED_RESULTS = []
# Absolute and relative tolerance to use when comparing to EXPECTED_RESULTS
__C.EXPECTED_RESULTS_RTOL = 0.1
__C.EXPECTED_RESULTS_ATOL = 0.005
# Set to send email in case of an EXPECTED_RESULTS failure
__C.EXPECTED_RESULTS_EMAIL = b''

# Models and proposals referred to by URL are downloaded to a local cache
# specified by DOWNLOAD_CACHE
__C.DOWNLOAD_CACHE = b'/tmp/detectron-download-cache'


# ---------------------------------------------------------------------------- #
# Cluster options
# ---------------------------------------------------------------------------- #
__C.CLUSTER = AttrDict()

# Flag to indicate if the code is running in a cluster environment
__C.CLUSTER.ON_CLUSTER = False


# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    (
        'FINAL_MSG',
        'MODEL.DILATION',
        'ROOT_GPU_ID',
        'RPN.ON',
        'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
        'TRAIN.DROPOUT',
        'USE_GPU_NMS',
        'TEST.NUM_TEST_IMAGES',
    )
)

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'MODEL.PS_GRID_SIZE': 'RFCN.PS_GRID_SIZE',
    'MODEL.ROI_HEAD': 'FAST_RCNN.ROI_BOX_HEAD',
    'MRCNN.MASK_HEAD_NAME': 'MRCNN.ROI_MASK_HEAD',
    'TRAIN.DATASET': (
        'TRAIN.DATASETS',
        "Also convert to a tuple, e.g., " +
        "'coco_2014_train' -> ('coco_2014_train',) or " +
        "'coco_2014_train:coco_2014_valminusminival' -> " +
        "('coco_2014_train', 'coco_2014_valminusminival')"
    ),
    'TRAIN.PROPOSAL_FILE': (
        'TRAIN.PROPOSAL_FILES',
        "Also convert to a tuple, e.g., " +
        "'path/to/file' -> ('path/to/file',) or " +
        "'path/to/file1:path/to/file2' -> " +
        "('path/to/file1', 'path/to/file2')"
    ),
}


def assert_and_infer_cfg(cache_urls=True):
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if cache_urls:
        cache_cfg_urls()


def cache_cfg_urls():
    """Download URLs in the config, cache them locally, and rewrite cfg to make
    use of the locally cached file.
    """
    __C.TRAIN.WEIGHTS = cache_url(__C.TRAIN.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TEST.WEIGHTS = cache_url(__C.TEST.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TRAIN.PROPOSAL_FILES = tuple(
        [cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TRAIN.PROPOSAL_FILES]
    )
    __C.TEST.PROPOSAL_FILES = tuple(
        [cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TEST.PROPOSAL_FILES]
    )


def get_output_dir(training=True):
    """Get the output directory determined by the current global config."""
    dataset = __C.TRAIN.DATASETS if training else __C.TEST.DATASETS
    dataset = ':'.join(dataset)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
