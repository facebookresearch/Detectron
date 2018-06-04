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

import logging
import numpy as np
import os
import shutil
import tempfile

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.datasets.roidb import combined_roidb_for_training
from detectron.modeling import model_builder
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import detectron.utils.net as nu

c2_utils.import_detectron_ops()


def get_params(model):
    blobs = {}  # gpu_0 blobs with unscoped_name as key
    all_blobs = {}  # all blobs with scoped name as key
    # Save all parameters
    for param in model.params:
        scoped_name = str(param)
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        if 'gpu_0' in scoped_name:
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
        all_blobs[scoped_name] = workspace.FetchBlob(scoped_name)
    for param in model.TrainableParams():
        scoped_name = str(param) + '_momentum'
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        if 'gpu_0' in scoped_name:
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
        all_blobs[scoped_name] = workspace.FetchBlob(scoped_name)
    return blobs, all_blobs


def add_momentum_init_ops(model):
    for param in model.TrainableParams(gpu_id=0):
        model.param_init_net.GaussianFill(
            [param + '_momentum'], param + '_momentum', mean=0.0, std=1.0)


def init_weights(model):
    # init weights in gpu_id = 0 and then broadcast
    workspace.RunNetOnce(model.param_init_net)
    nu.broadcast_parameters(model)


def test_restore_checkpoint():
    # Create Model
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    add_momentum_init_ops(model)
    init_weights(model)
    # Fill input blobs
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    model_builder.add_training_inputs(model, roidb=roidb)
    workspace.CreateNet(model.net)
    # Bookkeeping for checkpoint creation
    iter_num = 0
    checkpoints = {}
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    chk_file_path = os.path.join(output_dir, 'model_iter{}.pkl'.format(iter_num))
    checkpoints[iter_num] = chk_file_path
    # Save model weights
    nu.save_model_to_weights_file(checkpoints[iter_num], model)
    orig_gpu_0_params, orig_all_params = get_params(model)
    # Change the model weights
    init_weights(model)
    # Reload the weights in the model
    nu.initialize_gpu_from_weights_file(model, chk_file_path, gpu_id=0)
    nu.broadcast_parameters(model)
    shutil.rmtree(cfg.OUTPUT_DIR)
    _, restored_all_params = get_params(model)
    # Check if all params are loaded correctly
    for scoped_name, blob in orig_all_params.items():
        np.testing.assert_array_equal(blob, restored_all_params[scoped_name])
    # Check if broadcast_parameters works
    for scoped_name, blob in restored_all_params.items():
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        np.testing.assert_array_equal(blob, orig_gpu_0_params[unscoped_name])


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    np.random.seed(cfg.RNG_SEED)
    output_dir = tempfile.mkdtemp()
    # Generate config for test
    cfg.MODEL.TYPE = 'generalized_rcnn'
    cfg.MODEL.CONV_BODY = 'FPN.add_fpn_ResNet50_conv5_body'
    cfg.MODEL.NUM_CLASSES = 81
    cfg.MODEL.FASTER_RCNN = True
    cfg.FPN.FPN_ON = True
    cfg.FPN.MULTILEVEL_ROIS = True
    cfg.FPN.MULTILEVEL_RPN = True
    cfg.FAST_RCNN.ROI_BOX_HEAD = 'fast_rcnn_heads.add_roi_2mlp_head'
    cfg.FAST_RCNN.ROI_XFORM_METHOD = 'RoIAlign'
    cfg.OUTPUT_DIR = output_dir
    cfg.TRAIN.DATASETS = ('coco_2014_minival',)
    cfg.TRAIN.WEIGHTS = b''
    for num_gpu in range(workspace.NumCudaDevices()):
        cfg.immutable(False)
        cfg.NUM_GPUS = num_gpu + 1
        assert_and_infer_cfg()
        test_restore_checkpoint()
