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

# Adapted from https://github.com/caffe2/caffe2/blob/master/cmake/Summary.cmake

# Prints configuration summary.
function (detectron_print_config_summary)
  message(STATUS "Summary:")
  message(STATUS "  CMake version        : ${CMAKE_VERSION}")
  message(STATUS "  CMake command        : ${CMAKE_COMMAND}")
  message(STATUS "  System name          : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler         : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  CXX flags            : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Caffe2 version       : ${CAFFE2_VERSION}")
  message(STATUS "  Caffe2 include path  : ${CAFFE2_INCLUDE_DIRS}")
  message(STATUS "  Have CUDA            : ${HAVE_CUDA}")
  if (${HAVE_CUDA})
    message(STATUS "    CUDA version       : ${CUDA_VERSION}")
    message(STATUS "    CuDNN version      : ${CUDNN_VERSION}")
  endif()
endfunction()
