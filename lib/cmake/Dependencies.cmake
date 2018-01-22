# Adapted from https://github.com/caffe2/caffe2/blob/master/cmake/Dependencies.cmake

# Find the Caffe2 package.
# Caffe2 exports the required targets, so find_package should work for
# the standard Caffe2 installation. If you encounter problems with finding
# the Caffe2 package, make sure you have run `make install` when installing
# Caffe2 (`make install` populates your share/cmake/Caffe2).
find_package(Caffe2 REQUIRED)

# Find CUDA.
include(cmake/Cuda.cmake)
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
