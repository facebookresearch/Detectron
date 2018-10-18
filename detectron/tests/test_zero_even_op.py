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
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils


class ZeroEvenOpTest(unittest.TestCase):

    def _run_zero_even_op(self, X):
        op = core.CreateOperator('ZeroEven', ['X'], ['Y'])
        workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def _run_zero_even_op_gpu(self, X):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('ZeroEven', ['X'], ['Y'])
            workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_throws_on_non_1D_arrays(self):
        X = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegexp(RuntimeError, 'X\.ndim\(\) == 1'):
            self._run_zero_even_op(X)

    def test_handles_empty_arrays(self):
        X = np.array([], dtype=np.float32)
        Y_exp = np.copy(X)
        Y_act = self._run_zero_even_op(X)
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_sets_vals_at_even_inds_to_zero(self):
        X = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        Y_exp = np.array([0, 1, 0, 3, 0], dtype=np.float32)
        Y_act = self._run_zero_even_op(X)
        np.testing.assert_allclose(Y_act[0::2], Y_exp[0::2])

    def test_preserves_vals_at_odd_inds(self):
        X = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        Y_exp = np.array([0, 1, 0, 3, 0], dtype=np.float32)
        Y_act = self._run_zero_even_op(X)
        np.testing.assert_allclose(Y_act[1::2], Y_exp[1::2])

    def test_handles_even_length_arrays(self):
        X = np.random.rand(64).astype(np.float32)
        Y_exp = np.copy(X)
        Y_exp[0::2] = 0.0
        Y_act = self._run_zero_even_op(X)
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_handles_odd_length_arrays(self):
        X = np.random.randn(77).astype(np.float32)
        Y_exp = np.copy(X)
        Y_exp[0::2] = 0.0
        Y_act = self._run_zero_even_op(X)
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_gpu_throws_on_non_1D_arrays(self):
        X = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegexp(RuntimeError, 'X\.ndim\(\) == 1'):
            self._run_zero_even_op_gpu(X)

    def test_gpu_handles_empty_arrays(self):
        X = np.array([], dtype=np.float32)
        Y_exp = np.copy(X)
        Y_act = self._run_zero_even_op_gpu(X)
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_gpu_sets_vals_at_even_inds_to_zero(self):
        X = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        Y_exp = np.array([0, 1, 0, 3, 0], dtype=np.float32)
        Y_act = self._run_zero_even_op_gpu(X)
        np.testing.assert_allclose(Y_act[0::2], Y_exp[0::2])

    def test_gpu_preserves_vals_at_odd_inds(self):
        X = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        Y_exp = np.array([0, 1, 0, 3, 0], dtype=np.float32)
        Y_act = self._run_zero_even_op_gpu(X)
        np.testing.assert_allclose(Y_act[1::2], Y_exp[1::2])

    def test_gpu_handles_even_length_arrays(self):
        X = np.random.rand(64).astype(np.float32)
        Y_exp = np.copy(X)
        Y_exp[0::2] = 0.0
        Y_act = self._run_zero_even_op_gpu(X)
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_gpu_handles_odd_length_arrays(self):
        X = np.random.randn(77).astype(np.float32)
        Y_exp = np.copy(X)
        Y_exp[0::2] = 0.0
        Y_act = self._run_zero_even_op_gpu(X)
        np.testing.assert_allclose(Y_act, Y_exp)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_custom_ops()
    assert 'ZeroEven' in workspace.RegisteredOperators()
    unittest.main()
