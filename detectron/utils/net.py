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

"""Helper functions for working with Caffe2 networks (i.e., operator graphs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import logging
import numpy as np
import os
import pprint

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.utils.io import load_object
from detectron.utils.io import save_object
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_from_weights_file(model, weights_file, broadcast=True):
    """Initialize a model from weights stored in a pickled dictionary. If
    multiple GPUs are used, the loaded weights are synchronized on all GPUs,
    unless 'broadcast' is False.
    """
    initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    if broadcast:
        broadcast_parameters(model)


def initialize_gpu_from_weights_file(model, weights_file, gpu_id=0):
    """Initialize a network with ops on a specific GPU.

    If you use CUDA_VISIBLE_DEVICES to target specific GPUs, Caffe2 will
    automatically map logical GPU ids (starting from 0) to the physical GPUs
    specified in CUDA_VISIBLE_DEVICES.
    """
    logger.info('Loading weights from: {}'.format(weights_file))
    ws_blobs = workspace.Blobs()
    src_blobs = load_object(weights_file)

    if 'cfg' in src_blobs:
        saved_cfg = load_cfg(src_blobs['cfg'])
        configure_bbox_reg_weights(model, saved_cfg)
    if 'blobs' in src_blobs:
        # Backwards compat--dictionary used to be only blobs, now they are
        # stored under the 'blobs' key
        src_blobs = src_blobs['blobs']
    # Initialize weights on GPU gpu_id only
    unscoped_param_names = OrderedDict()  # Print these out in model order
    for blob in model.params:
        unscoped_param_names[c2_utils.UnscopeName(str(blob))] = True
    with c2_utils.NamedCudaScope(gpu_id):
        for unscoped_param_name in unscoped_param_names.keys():
            if (unscoped_param_name.find(']_') >= 0 and
                    unscoped_param_name not in src_blobs):
                # Special case for sharing initialization from a pretrained
                # model:
                # If a blob named '_[xyz]_foo' is in model.params and not in
                # the initialization blob dictionary, then load source blob
                # 'foo' into destination blob '_[xyz]_foo'
                src_name = unscoped_param_name[
                    unscoped_param_name.find(']_') + 2:]
            else:
                src_name = unscoped_param_name
            if src_name not in src_blobs:
                logger.info('{:s} not found'.format(src_name))
                continue
            dst_name = core.ScopedName(unscoped_param_name)
            has_momentum = src_name + '_momentum' in src_blobs
            has_momentum_str = ' [+ momentum]' if has_momentum else ''
            logger.info(
                '{:s}{:} loaded from weights file into {:s}: {}'.format(
                    src_name, has_momentum_str, dst_name, src_blobs[src_name]
                    .shape
                )
            )
            if dst_name in ws_blobs:
                # If the blob is already in the workspace, make sure that it
                # matches the shape of the loaded blob
                ws_blob = workspace.FetchBlob(dst_name)
                assert ws_blob.shape == src_blobs[src_name].shape, \
                    ('Workspace blob {} with shape {} does not match '
                     'weights file shape {}').format(
                        src_name,
                        ws_blob.shape,
                        src_blobs[src_name].shape)
            workspace.FeedBlob(
                dst_name,
                src_blobs[src_name].astype(np.float32, copy=False))
            if has_momentum:
                workspace.FeedBlob(
                    dst_name + '_momentum',
                    src_blobs[src_name + '_momentum'].astype(
                        np.float32, copy=False))

    # We preserve blobs that are in the weights file but not used by the current
    # model. We load these into CPU memory under the '__preserve__/' namescope.
    # These blobs will be stored when saving a model to a weights file. This
    # feature allows for alternating optimization of Faster R-CNN in which blobs
    # unused by one step can still be preserved forward and used to initialize
    # another step.
    for src_name in src_blobs.keys():
        if (src_name not in unscoped_param_names and
                not src_name.endswith('_momentum') and
                src_blobs[src_name] is not None):
            with c2_utils.CpuScope():
                workspace.FeedBlob(
                    '__preserve__/{:s}'.format(src_name), src_blobs[src_name])
                logger.info(
                    '{:s} preserved in workspace (unused)'.format(src_name))


def save_model_to_weights_file(weights_file, model):
    """Stash model weights in a dictionary and pickle them to a file. We map
    GPU device scoped names to unscoped names (e.g., 'gpu_0/conv1_w' ->
    'conv1_w').
    """
    logger.info(
        'Saving parameters and momentum to {}'.format(
            os.path.abspath(weights_file)))
    blobs = {}
    # Save all parameters
    for param in model.params:
        scoped_name = str(param)
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        if unscoped_name not in blobs:
            logger.debug(' {:s} -> {:s}'.format(scoped_name, unscoped_name))
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save momentum
    for param in model.TrainableParams():
        scoped_name = str(param) + '_momentum'
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        if unscoped_name not in blobs:
            logger.debug(' {:s} -> {:s}'.format(scoped_name, unscoped_name))
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save preserved blobs
    for scoped_name in workspace.Blobs():
        if scoped_name.startswith('__preserve__/'):
            unscoped_name = c2_utils.UnscopeName(scoped_name)
            if unscoped_name not in blobs:
                logger.debug(
                    ' {:s} -> {:s} (preserved)'.format(
                        scoped_name, unscoped_name))
                blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    cfg_yaml = envu.yaml_dump(cfg)
    save_object(dict(blobs=blobs, cfg=cfg_yaml), weights_file)


def broadcast_parameters(model):
    """Copy parameter blobs from GPU 0 over the corresponding parameter blobs
    on GPUs 1 through cfg.NUM_GPUS - 1.
    """
    if cfg.NUM_GPUS == 1:
        # no-op if only running on a single GPU
        return

    def _do_broadcast(all_blobs):
        assert len(all_blobs) % cfg.NUM_GPUS == 0, \
            ('Unexpected value for NUM_GPUS. Make sure you are not '
             'running single-GPU inference with NUM_GPUS > 1.')
        blobs_per_gpu = int(len(all_blobs) / cfg.NUM_GPUS)
        for i in range(blobs_per_gpu):
            blobs = [p for p in all_blobs[i::blobs_per_gpu]]
            data = workspace.FetchBlob(blobs[0])
            logger.debug('Broadcasting {} to'.format(str(blobs[0])))
            for i, p in enumerate(blobs[1:]):
                logger.debug(' |-> {}'.format(str(p)))
                with c2_utils.CudaScope(i + 1):
                    workspace.FeedBlob(p, data)

    _do_broadcast(model.params)
    _do_broadcast([b + '_momentum' for b in model.TrainableParams()])


def sum_multi_gpu_blob(blob_name):
    """Return the sum of a scalar blob held on multiple GPUs."""
    val = 0
    for i in range(cfg.NUM_GPUS):
        val += float(workspace.FetchBlob('gpu_{}/{}'.format(i, blob_name)))
    return val


def average_multi_gpu_blob(blob_name):
    """Return the average of a scalar blob held on multiple GPUs."""
    return sum_multi_gpu_blob(blob_name) / cfg.NUM_GPUS


def print_net(model, namescope='gpu_0'):
    """Print the model network."""
    logger.info('Printing model: {}'.format(model.net.Name()))
    op_list = model.net.Proto().op
    for op in op_list:
        input_name = op.input
        # For simplicity: only print the first output
        # Not recommended if there are split layers
        output_name = str(op.output[0])
        op_type = op.type
        op_name = op.name

        if namescope is None or output_name.startswith(namescope):
            # Only print the forward pass network
            if output_name.find('grad') >= 0 or output_name.find('__m') >= 0:
                continue

            try:
                # Under some conditions (e.g., dynamic memory optimization)
                # it is possible that the network frees some blobs when they are
                # no longer needed. Handle this case...
                output_shape = workspace.FetchBlob(output_name).shape
            except BaseException:
                output_shape = '<unknown>'

            first_blob = True
            op_label = op_type + (op_name if op_name == '' else ':' + op_name)
            suffix = ' ------- (op: {})'.format(op_label)
            for j in range(len(input_name)):
                if input_name[j] in model.params:
                    continue
                input_blob = workspace.FetchBlob(input_name[j])
                if isinstance(input_blob, np.ndarray):
                    input_shape = input_blob.shape
                    logger.info('{:28s}: {:20s} => {:28s}: {:20s}{}'.format(
                        c2_utils.UnscopeName(str(input_name[j])),
                        '{}'.format(input_shape),
                        c2_utils.UnscopeName(str(output_name)),
                        '{}'.format(output_shape),
                        suffix))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info('End of model: {}'.format(model.net.Name()))


def configure_bbox_reg_weights(model, saved_cfg):
    """Compatibility for old models trained with bounding box regression
    mean/std normalization (instead of fixed weights).
    """
    if 'MODEL' not in saved_cfg or 'BBOX_REG_WEIGHTS' not in saved_cfg.MODEL:
        logger.warning('Model from weights file was trained before config key '
                       'MODEL.BBOX_REG_WEIGHTS was added. Forcing '
                       'MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.) to ensure '
                       'correct **inference** behavior.')
        # Generally we don't allow modifying the config, but this is a one-off
        # hack to support some very old models
        is_immutable = cfg.is_immutable()
        cfg.immutable(False)
        cfg.MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.)
        cfg.immutable(is_immutable)
        logger.info('New config:')
        logger.info(pprint.pformat(cfg))
        assert not model.train, (
            'This model was trained with an older version of the code that '
            'used bounding box regression mean/std normalization. It can no '
            'longer be used for training. To upgrade it to a trainable model '
            'please use fb/compat/convert_bbox_reg_normalized_model.py.'
        )


def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn
