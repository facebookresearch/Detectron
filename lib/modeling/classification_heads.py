# Copyright (c) 2017-present, Sterblue, Inc.
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

"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import utils.blob as blob_utils




def add_mlp_outputs(model, blob_in, dim):
    """Add  classification  ops."""

    bb,dimO = add_2mlp_head(model, blob_in, dim)
    model.FC(
        bb,
        'logits',
        dimO,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_prob', engine='CUDNN')

def add_mlp_losses(model):
    
    """Add losses for classification """

    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['logits', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients



def add_1mlp_head(model, blob_in, dim_in):

    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(blob_in, 'fc6', dim_in , hidden_dim)
    model.Relu('fc6', 'fc6')

    return 'fc6', hidden_dim

def add_2mlp_head(model, blob_in, dim_in):

    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(blob_in, 'fc6', dim_in , hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim , hidden_dim)
    model.Relu('fc7', 'fc7')

    return 'fc7', hidden_dim
