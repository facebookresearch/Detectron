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

# Adapted from https://github.com/caffe2/caffe2/blob/master/cmake/Dependencies.cmake

# Find CUDA.
include(cmake/legacy/Cuda.cmake)
if (HAVE_CUDA)
  # CUDA 9.x requires GCC version <= 6
  if ((CUDA_VERSION VERSION_EQUAL   9.0) OR
      (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 10.0))
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 7.0 AND
        CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
      message(FATAL_ERROR
        "CUDA ${CUDA_VERSION} is not compatible with GCC version >= 7. "
        "Use the following option to use another version (for example): \n"
        "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-6\n")
    endif()
  # CUDA 8.0 requires GCC version <= 5
  elseif (CUDA_VERSION VERSION_EQUAL 8.0)
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
        CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
      message(FATAL_ERROR
        "CUDA 8.0 is not compatible with GCC version >= 6. "
        "Use the following option to use another version (for example): \n"
        "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
    endif()
  endif()
endif()

# Find CUDNN.
if (HAVE_CUDA)
  find_package(CuDNN REQUIRED)
  if (CUDNN_FOUND)
    caffe2_include_directories(${CUDNN_INCLUDE_DIRS})
  endif()
endif()
