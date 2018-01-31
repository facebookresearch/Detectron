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

"""Optimization operator graph construction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from caffe2.python import muji

from core.config import cfg
import utils.c2 as c2_utils

logger = logging.getLogger(__name__)


def build_data_parallel_model(model, single_gpu_build_func):
    """Build a data parallel model given a function that builds the model on a
    single GPU.
    """
    if model.only_build_forward_pass:
        single_gpu_build_func(model)
    elif model.train:
        all_loss_gradients = _build_forward_graph(model, single_gpu_build_func)
        # Add backward pass on all GPUs
        model.AddGradientOperators(all_loss_gradients)
        if cfg.NUM_GPUS > 1:
            _add_allreduce_graph(model)
        for gpu_id in range(cfg.NUM_GPUS):
            # After allreduce, all GPUs perform SGD updates on their identical
            # params and gradients in parallel
            with c2_utils.NamedCudaScope(gpu_id):
                add_single_gpu_param_update_ops(model, gpu_id)
    else:
        # Test-time network operates on single GPU
        # Test-time parallelism is implemented through multiprocessing
        with c2_utils.NamedCudaScope(model.target_gpu_id):
            single_gpu_build_func(model)


def _build_forward_graph(model, single_gpu_build_func):
    """Construct the forward graph on each GPU."""
    all_loss_gradients = {}  # Will include loss gradients from all GPUs
    # Build the model on each GPU with correct name and device scoping
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils.NamedCudaScope(gpu_id):
            all_loss_gradients.update(single_gpu_build_func(model))
    return all_loss_gradients


def _add_allreduce_graph(model):
    """Construct the graph that performs Allreduce on the gradients."""
    # Need to all-reduce the per-GPU gradients if training with more than 1 GPU
    all_params = model.TrainableParams()
    assert len(all_params) % cfg.NUM_GPUS == 0
    # The model parameters are replicated on each GPU, get the number
    # distinct parameter blobs (i.e., the number of parameter blobs on
    # each GPU)
    params_per_gpu = int(len(all_params) / cfg.NUM_GPUS)
    with c2_utils.CudaScope(0):
        # Iterate over distinct parameter blobs
        for i in range(params_per_gpu):
            # Gradients from all GPUs for this parameter blob
            gradients = [
                model.param_to_grad[p] for p in all_params[i::params_per_gpu]
            ]
            if len(gradients) > 0:
                if cfg.USE_NCCL:
                    model.net.NCCLAllreduce(gradients, gradients)
                else:
                    muji.Allreduce(model.net, gradients, reduced_affix='')


def add_single_gpu_param_update_ops(model, gpu_id):
    # Learning rate of 0 is a dummy value to be set properly at the
    # start of training
    lr = model.param_init_net.ConstantFill(
        [], 'lr', shape=[1], value=0.0
    )
    one = model.param_init_net.ConstantFill(
        [], 'one', shape=[1], value=1.0
    )
    wd = model.param_init_net.ConstantFill(
        [], 'wd', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY
    )
    for param in model.TrainableParams(gpu_id=gpu_id):
        logger.debug('param ' + str(param) + ' will be updated')
        param_grad = model.param_to_grad[param]
        # Initialize momentum vector
        param_momentum = model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )
        if param in model.biases:
            # Special treatment for biases (mainly to match historical impl.
            # details):
            # (1) Do not apply weight decay
            # (2) Use a 2x higher learning rate
            model.Scale(param_grad, param_grad, scale=2.0)
        elif cfg.SOLVER.WEIGHT_DECAY > 0:
            # Apply weight decay to non-bias weights
            model.WeightedSum([param_grad, one, param, wd], param_grad)
        # Update param_grad and param_momentum in place
        model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, lr, param],
            [param_grad, param_momentum, param],
            momentum=cfg.SOLVER.MOMENTUM
        )
