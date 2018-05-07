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

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='detectron.utils.cython_bbox',
        sources=[
            'detectron/utils/cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
    Extension(
        name='detectron.utils.cython_nms',
        sources=[
            'detectron/utils/cython_nms.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )
]

setup(
    name='Detectron',
    packages=['detectron'],
    ext_modules=cythonize(ext_modules)
)
