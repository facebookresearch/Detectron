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

"""Environment helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

# Default value of the CMake install prefix
_CMAKE_INSTALL_PREFIX = '/usr/local'


def get_runtime_dir():
    """Retrieve the path to the runtime directory."""
    return sys.path[0]


def get_py_bin_ext():
    """Retrieve python binary extension."""
    return '.py'


def set_up_matplotlib():
    """Set matplotlib up."""
    import matplotlib
    # Use a non-interactive backend
    matplotlib.use('Agg')


def exit_on_error():
    """Exit from a detectron tool when there's an error."""
    sys.exit(1)


def import_nccl_ops():
    """Import NCCL ops."""
    # There is no need to load NCCL ops since the
    # NCCL dependency is built into the Caffe2 gpu lib
    pass


def get_detectron_ops_lib():
    """Retrieve Detectron ops library."""
    # Candidate prefixes for the detectron ops lib path
    prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix] + sys.path
    # Search for detectron ops lib
    for prefix in prefixes:
        ops_path = os.path.join(prefix, 'lib/libcaffe2_detectron_ops_gpu.so')
        if os.path.exists(ops_path):
            # TODO(ilijar): Switch to using a logger
            print('Found Detectron ops lib: {}'.format(ops_path))
            break
    assert os.path.exists(ops_path), \
        ('Detectron ops lib not found; make sure that your Caffe2 '
         'version includes Detectron module')
    return ops_path


def get_custom_ops_lib():
    """Retrieve custom ops library."""
    det_dir, _ = os.path.split(os.path.dirname(__file__))
    root_dir, _ = os.path.split(det_dir)
    custom_ops_lib = os.path.join(
        root_dir, 'build/libcaffe2_detectron_custom_ops_gpu.so')
    assert os.path.exists(custom_ops_lib), \
        'Custom ops lib not found at \'{}\''.format(custom_ops_lib)
    return custom_ops_lib
