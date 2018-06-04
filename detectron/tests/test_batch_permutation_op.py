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
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import gradient_checker
from caffe2.python import workspace

import detectron.utils.logging as logging_utils
import detectron.utils.c2 as c2_utils


class BatchPermutationOpTest(unittest.TestCase):
    def _run_op_test(self, X, I, check_grad=False):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('BatchPermutation', ['X', 'I'], ['Y'])
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('I', I)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')

        if check_grad:
            gc = gradient_checker.GradientChecker(
                stepsize=0.1,
                threshold=0.001,
                device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
            )

            res, grad, grad_estimated = gc.CheckSimple(op, [X, I], 0, [0])
            self.assertTrue(res, 'Grad check failed')

        Y_ref = X[I]
        np.testing.assert_allclose(Y, Y_ref, rtol=1e-5, atol=1e-08)

    def _run_speed_test(self, iters=5, N=1024):
        """This function provides an example of how to benchmark custom
        operators using the Caffe2 'prof_dag' network execution type. Please
        note that for 'prof_dag' to work, Caffe2 must be compiled with profiling
        support using the `-DUSE_PROF=ON` option passed to `cmake` when building
        Caffe2.
        """
        net = core.Net('test')
        net.Proto().type = 'prof_dag'
        net.Proto().num_workers = 2
        Y = net.BatchPermutation(['X', 'I'], 'Y')
        Y_flat = net.FlattenToVec([Y], 'Y_flat')
        loss = net.AveragedLoss([Y_flat], 'loss')
        net.AddGradientOperators([loss])
        workspace.CreateNet(net)

        X = np.random.randn(N, 256, 14, 14)
        for _i in range(iters):
            I = np.random.permutation(N)
            workspace.FeedBlob('X', X.astype(np.float32))
            workspace.FeedBlob('I', I.astype(np.int32))
            workspace.RunNet(net.Proto().name)
            np.testing.assert_allclose(
                workspace.FetchBlob('Y'), X[I], rtol=1e-5, atol=1e-08
            )

    def test_forward_and_gradient(self):
        A = np.random.randn(2, 3, 5, 7).astype(np.float32)
        I = np.array([0, 1], dtype=np.int32)
        self._run_op_test(A, I, check_grad=True)

        A = np.random.randn(2, 3, 5, 7).astype(np.float32)
        I = np.array([1, 0], dtype=np.int32)
        self._run_op_test(A, I, check_grad=True)

        A = np.random.randn(10, 3, 5, 7).astype(np.float32)
        I = np.array(np.random.permutation(10), dtype=np.int32)
        self._run_op_test(A, I, check_grad=True)

    def test_size_exceptions(self):
        A = np.random.randn(2, 256, 42, 86).astype(np.float32)
        I = np.array(np.random.permutation(10), dtype=np.int32)
        with self.assertRaises(RuntimeError):
            self._run_op_test(A, I)

    # See doc string in _run_speed_test
    # def test_perf(self):
    #     with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
    #         self._run_speed_test()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_detectron_ops()
    assert 'BatchPermutation' in workspace.RegisteredOperators()
    logging_utils.setup_logging(__name__)
    unittest.main()
