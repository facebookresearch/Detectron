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

import numpy as np
import logging
import unittest
import mock

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import muji
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.roi_data.loader import RoIDataLoader
import detectron.utils.logging as logging_utils


def get_roidb_blobs(roidb):
    blobs = {}
    blobs['data'] = np.stack([entry['data'] for entry in roidb])
    return blobs, True


def get_net(data_loader, name):
    logger = logging.getLogger(__name__)
    blob_names = data_loader.get_output_names()
    net = core.Net(name)
    net.type = 'dag'
    for gpu_id in range(cfg.NUM_GPUS):
        with core.NameScope('gpu_{}'.format(gpu_id)):
            with core.DeviceScope(muji.OnGPU(gpu_id)):
                for blob_name in blob_names:
                    blob = core.ScopedName(blob_name)
                    workspace.CreateBlob(blob)
                net.DequeueBlobs(
                    data_loader._blobs_queue_name, blob_names)
    logger.info("Protobuf:\n" + str(net.Proto()))

    return net


def get_roidb_sample_data(sample_data):
    roidb = []
    for _ in range(np.random.randint(4, 10)):
        roidb.append({'data': sample_data})
    return roidb


def create_loader_and_network(sample_data, name):
    roidb = get_roidb_sample_data(sample_data)
    loader = RoIDataLoader(roidb)
    net = get_net(loader, 'dequeue_net_train')
    loader.register_sigint_handler()
    loader.start(prefill=False)
    return loader, net


def run_net(net):
    workspace.RunNetOnce(net)
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, 0)
    name_scope = 'gpu_{}'.format(0)
    with core.NameScope(name_scope):
        with core.DeviceScope(gpu_dev):
            data = workspace.FetchBlob(core.ScopedName('data'))
            return data


class TestRoIDataLoader(unittest.TestCase):
    @mock.patch(
        'detectron.roi_data.loader.get_minibatch_blob_names',
        return_value=[u'data']
    )
    @mock.patch(
        'detectron.roi_data.loader.get_minibatch',
        side_effect=get_roidb_blobs
    )
    def test_two_parallel_loaders(self, _1, _2):
        train_data = np.random.rand(2, 3, 3).astype(np.float32)
        train_loader, train_net = create_loader_and_network(train_data,
                                                            'dequeue_net_train')
        test_data = np.random.rand(2, 4, 4).astype(np.float32)
        test_loader, test_net = create_loader_and_network(test_data,
                                                          'dequeue_net_test')
        for _ in range(5):
            data = run_net(train_net)
            self.assertEqual(data[0].tolist(), train_data.tolist())
            data = run_net(test_net)
            self.assertEqual(data[0].tolist(), test_data.tolist())
        test_loader.shutdown()
        train_loader.shutdown()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = logging_utils.setup_logging(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    np.random.seed(cfg.RNG_SEED)
    cfg.TRAIN.ASPECT_GROUPING = False
    cfg.NUM_GPUS = 2
    assert_and_infer_cfg()
    unittest.main()
