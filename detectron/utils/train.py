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

"""Utilities driving the train_net binary"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from shutil import copyfile
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import os
import re

from caffe2.python import memonger
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.datasets.roidb import combined_roidb_for_training
from detectron.modeling import model_builder
from detectron.utils import lr_policy
from detectron.utils.training_stats import TrainingStats
import detectron.utils.env as envu
import detectron.utils.net as nu


def train_model():
    """Model training loop."""
    model, weights_file, start_iter, checkpoints, output_dir = create_model()
    if 'final' in checkpoints:
        # The final model was found in the output directory, so nothing to do
        return checkpoints

    setup_model_for_training(model, weights_file, output_dir)
    training_stats = TrainingStats(model)
    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    for cur_iter in range(start_iter, cfg.SOLVER.MAX_ITER):
        if model.roi_data_loader.has_stopped():
            handle_critical_error(model, 'roi_data_loader failed')
        training_stats.IterTic()
        lr = model.UpdateWorkspaceLr(cur_iter, lr_policy.get_lr_at_iter(cur_iter))
        workspace.RunNet(model.net.Proto().name)
        if cur_iter == start_iter:
            nu.print_net(model)
        training_stats.IterToc()
        training_stats.UpdateIterStats()
        training_stats.LogIterStats(cur_iter, lr)

        if (cur_iter + 1) % CHECKPOINT_PERIOD == 0 and cur_iter > start_iter:
            checkpoints[cur_iter] = os.path.join(
                output_dir, 'model_iter{}.pkl'.format(cur_iter)
            )
            nu.save_model_to_weights_file(checkpoints[cur_iter], model)

        if cur_iter == start_iter + training_stats.LOG_PERIOD:
            # Reset the iteration timer to remove outliers from the first few
            # SGD iterations
            training_stats.ResetIterTimer()

        if np.isnan(training_stats.iter_total_loss):
            handle_critical_error(model, 'Loss is NaN')

    # Save the final model
    checkpoints['final'] = os.path.join(output_dir, 'model_final.pkl')
    nu.save_model_to_weights_file(checkpoints['final'], model)
    # Shutdown data loading threads
    model.roi_data_loader.shutdown()
    return checkpoints


def handle_critical_error(model, msg):
    logger = logging.getLogger(__name__)
    logger.critical(msg)
    model.roi_data_loader.shutdown()
    raise Exception(msg)


def create_model():
    """Build the model and look for saved model checkpoints in case we can
    resume from one.
    """
    logger = logging.getLogger(__name__)
    start_iter = 0
    checkpoints = {}
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    weights_file = cfg.TRAIN.WEIGHTS
    if cfg.TRAIN.AUTO_RESUME:
        # Check for the final model (indicates training already finished)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists; no need to train!')
            return None, None, None, {'final': final_path}, output_dir

        if cfg.TRAIN.COPY_WEIGHTS:
            copyfile(
                weights_file,
                os.path.join(output_dir, os.path.basename(weights_file)))
            logger.info('Copy {} to {}'.format(weights_file, output_dir))

        # Find the most recent checkpoint (highest iteration number)
        files = os.listdir(output_dir)
        for f in files:
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter > start_iter:
                    # Start one iteration immediately after the checkpoint iter
                    start_iter = checkpoint_iter + 1
                    resume_weights_file = f

        if start_iter > 0:
            # Override the initialization weights with the found checkpoint
            weights_file = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                format(weights_file, start_iter)
            )

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    if cfg.MEMONGER:
        optimize_memory(model)
    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)
    return model, weights_file, start_iter, checkpoints, output_dir


def optimize_memory(model):
    """Save GPU memory through blob sharing."""
    for device in range(cfg.NUM_GPUS):
        namescope = 'gpu_{}/'.format(device)
        losses = [namescope + l for l in model.losses]
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses,
            set(model.param_to_grad.values()),
            namescope,
            share_activations=cfg.MEMONGER_SHARE_ACTIVATIONS
        )


def setup_model_for_training(model, weights_file, output_dir):
    """Loaded saved weights and create the network in the C2 workspace."""
    logger = logging.getLogger(__name__)
    add_model_training_inputs(model)

    if weights_file:
        # Override random weight initialization with weights from a saved model
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs
    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)

    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
    dump_proto_files(model, output_dir)

    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    model.roi_data_loader.start(prefill=True)
    return output_dir


def add_model_training_inputs(model):
    """Load the training dataset and attach the training inputs to the model."""
    logger = logging.getLogger(__name__)
    logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASETS))
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    logger.info('{:d} roidb entries'.format(len(roidb)))
    model_builder.add_training_inputs(model, roidb=roidb)


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))
