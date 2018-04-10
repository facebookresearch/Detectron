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

"""VGG16 from https://arxiv.org/abs/1409.1556."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg

from caffe2.python import brew


def add_VGG16_conv5_body(model):
    brew.conv(model, 'data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    brew.relu(model, 'conv1_1', 'conv1_1')
    brew.conv(model, 'conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    brew.relu(model, 'conv1_2', 'conv1_2')
    brew.max_pool(model, 'conv1_2', 'pool1', kernel=2, pad=0, stride=2)
    brew.conv(model, 'pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    brew.relu(model, 'conv2_1', 'conv2_1')
    brew.conv(model, 'conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    brew.relu(model, 'conv2_2', 'conv2_2')
    brew.max_pool(model, 'conv2_2', 'pool2', kernel=2, pad=0, stride=2)
    model.StopGradient('pool2', 'pool2')
    brew.conv(model, 'pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    brew.relu(model, 'conv3_1', 'conv3_1')
    brew.conv(model, 'conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    brew.relu(model, 'conv3_2', 'conv3_2')
    brew.conv(model, 'conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    brew.relu(model, 'conv3_3', 'conv3_3')
    brew.max_pool(model, 'conv3_3', 'pool3', kernel=2, pad=0, stride=2)
    brew.conv(model, 'pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    brew.relu(model, 'conv4_1', 'conv4_1')
    brew.conv(model, 'conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    brew.relu(model, 'conv4_2', 'conv4_2')
    brew.conv(model, 'conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    brew.relu(model, 'conv4_3', 'conv4_3')
    brew.max_pool(model, 'conv4_3', 'pool4', kernel=2, pad=0, stride=2)
    brew.conv(model, 'pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)
    brew.relu(model, 'conv5_1', 'conv5_1')
    brew.conv(model, 'conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)
    brew.relu(model, 'conv5_2', 'conv5_2')
    brew.conv(model, 'conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)
    blob_out = brew.relu(model, 'conv5_3', 'conv5_3')
    return blob_out, 512, 1. / 16.


def add_VGG16_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    brew.fc(model, 'pool5', 'fc6', dim_in * 7 * 7, 4096)
    brew.relu(model, 'fc6', 'fc6')
    brew.fc(model, 'fc6', 'fc7', 4096, 4096)
    blob_out = brew.relu(model, 'fc7', 'fc7')
    return blob_out, 4096
