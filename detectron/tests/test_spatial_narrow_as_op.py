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

import detectron.utils.c2 as c2_utils
import detectron.utils.logging as logging_utils


class SpatialNarrowAsOpTest(unittest.TestCase):
    def _run_test(self, A, B, check_grad=False):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('SpatialNarrowAs', ['A', 'B'], ['C'])
            workspace.FeedBlob('A', A)
            workspace.FeedBlob('B', B)
        workspace.RunOperatorOnce(op)
        C = workspace.FetchBlob('C')

        if check_grad:
            gc = gradient_checker.GradientChecker(
                stepsize=0.005,
                threshold=0.005,
                device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
            )

            res, grad, grad_estimated = gc.CheckSimple(op, [A, B], 0, [0])
            self.assertTrue(res, 'Grad check failed')

        dims = C.shape
        C_ref = A[:dims[0], :dims[1], :dims[2], :dims[3]]
        np.testing.assert_allclose(C, C_ref, rtol=1e-5, atol=1e-08)

    def test_small_forward_and_gradient(self):
        A = np.random.randn(2, 3, 5, 7).astype(np.float32)
        B = np.random.randn(2, 3, 2, 2).astype(np.float32)
        self._run_test(A, B, check_grad=True)

        A = np.random.randn(2, 3, 5, 7).astype(np.float32)
        B = np.random.randn(2, 3, 5).astype(np.float32)
        self._run_test(A, B, check_grad=True)

    def test_large_forward(self):
        A = np.random.randn(2, 256, 42, 100).astype(np.float32)
        B = np.random.randn(2, 256, 35, 87).astype(np.float32)
        self._run_test(A, B)

        A = np.random.randn(2, 256, 42, 87).astype(np.float32)
        B = np.random.randn(2, 256, 35, 87).astype(np.float32)
        self._run_test(A, B)

    def test_size_exceptions(self):
        A = np.random.randn(2, 256, 42, 86).astype(np.float32)
        B = np.random.randn(2, 256, 35, 87).astype(np.float32)
        with self.assertRaises(RuntimeError):
            self._run_test(A, B)

        A = np.random.randn(2, 255, 42, 88).astype(np.float32)
        B = np.random.randn(2, 256, 35, 87).astype(np.float32)
        with self.assertRaises(RuntimeError):
            self._run_test(A, B)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_detectron_ops()
    assert 'SpatialNarrowAs' in workspace.RegisteredOperators()
    logging_utils.setup_logging(__name__)
    unittest.main()
