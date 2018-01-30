#!/usr/bin/env python2

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

"""Train a network with Detectron."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import os
import pprint
import re
import sys
import test_net

from caffe2.python import memonger
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import get_output_dir
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from datasets.roidb import combined_roidb_for_training
from modeling import model_builder
from utils import lr_policy
from utils.logging import setup_logging
from utils.training_stats import TrainingStats
import utils.c2
import utils.env as envu
import utils.net as nu

utils.c2.import_contrib_ops()
utils.c2.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Initialize C2
    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
    )
    # Set up logging and load config options
    logger = setup_logging(__name__)
    logging.getLogger('roi_data.loader').setLevel(logging.INFO)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)
    # Execute the training run
    checkpoints = train_model()
    # Test the trained model
    if not args.skip_test:
        test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)


def train_model():
    """Model training loop."""
    logger = logging.getLogger(__name__)
    model, start_iter, checkpoints, output_dir = create_model()
    if 'final' in checkpoints:
        # The final model was found in the output directory, so nothing to do
        return checkpoints

    setup_model_for_training(model, output_dir)
    training_stats = TrainingStats(model)
    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    for cur_iter in range(start_iter, cfg.SOLVER.MAX_ITER):
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
            logger.critical('Loss is NaN, exiting...')
            model.roi_data_loader.shutdown()
            envu.exit_on_error()

    # Save the final model
    checkpoints['final'] = os.path.join(output_dir, 'model_final.pkl')
    nu.save_model_to_weights_file(checkpoints['final'], model)
    # Shutdown data loading threads
    model.roi_data_loader.shutdown()
    return checkpoints


def create_model():
    """Build the model and look for saved model checkpoints in case we can
    resume from one.
    """
    logger = logging.getLogger(__name__)
    start_iter = 0
    checkpoints = {}
    output_dir = get_output_dir(training=True)
    if cfg.TRAIN.AUTO_RESUME:
        # Check for the final model (indicates training already finished)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists; no need to train!')
            return None, None, {'final': final_path}, output_dir

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
            cfg.TRAIN.WEIGHTS = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                format(cfg.TRAIN.WEIGHTS, start_iter)
            )

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    if cfg.MEMONGER:
        optimize_memory(model)
    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)
    return model, start_iter, checkpoints, output_dir


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


def setup_model_for_training(model, output_dir):
    """Loaded saved weights and create the network in the C2 workspace."""
    logger = logging.getLogger(__name__)
    add_model_training_inputs(model)

    if cfg.TRAIN.WEIGHTS:
        # Override random weight initialization with weights from a saved model
        nu.initialize_gpu_from_weights_file(model, cfg.TRAIN.WEIGHTS, gpu_id=0)
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


def test_model(model_file, multi_gpu_testing, opts=None):
    """Test a model."""
    # All arguments to inference functions are passed via cfg
    cfg.TEST.WEIGHTS = model_file
    # Clear memory before inference
    workspace.ResetWorkspace()
    # Run inference
    test_net.main(multi_gpu_testing=multi_gpu_testing)


if __name__ == '__main__':
    main()
