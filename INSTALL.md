# Installing Detectron

This document covers how to install Detectron, its dependencies (including Caffe2), and the COCO dataset.

- For general information about Detectron, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Detectron operators currently do not have CPU implementation; a GPU system is required.
- Detectron has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.

## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install Python dependencies:

```
pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy
```

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## Detectron

Clone the Detectron repository:

```
# DETECTRON=/path/to/clone/detectron
git clone https://github.com/facebookresearch/detectron $DETECTRON
```

Set up Python modules:

```
cd $DETECTRON/lib && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](tests/test_spatial_narrow_as_op.py)):

```
python2 $DETECTRON/tests/test_spatial_narrow_as_op.py
```

## That's All You Need for Inference

At this point, you can run inference using pretrained Detectron models. Take a look at our [inference tutorial](GETTING_STARTED.md) for an example. If you want to train models on the COCO dataset, then please continue with the installation instructions.

## Datasets

Detectron finds datasets via symlinks from `lib/datasets/data` to the actual locations where the dataset images and annotations are stored. For instructions on how to create symlinks for COCO and other datasets, please see [`lib/datasets/data/README.md`](lib/datasets/data/README.md).

After symlinks have been created, that's all you need to start training models.

## Advanced Topic: Custom Operators for New Research Projects

Please read the custom operators section of the [`FAQ`](FAQ.md) first.

For convenience, we provide CMake support for building custom operators. All custom operators are built into a single library that can be loaded dynamically from Python.
Place your custom operator implementation under [`lib/ops/`](lib/ops/) and see [`tests/test_zero_even_op.py`](tests/test_zero_even_op.py) for an example of how to load custom operators from Python.

Build the custom operators library:

```
cd $DETECTRON/lib && make ops
```

Check that the custom operator tests pass:

```
python2 $DETECTRON/tests/test_zero_even_op.py
```

## Docker Image

We provide a [`Dockerfile`](docker/Dockerfile) that you can use to build a Detectron image on top of a Caffe2 image that satisfies the requirements outlined at the top. If you would like to use a Caffe2 image different from the one we use by default, please make sure that it includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).

Build the image:

```
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .
```

Run the image (e.g. for [`BatchPermutationOp test`](tests/test_batch_permutation_op.py)):

```
nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python2 tests/test_batch_permutation_op.py
```

## Troubleshooting

In case of Caffe2 installation problems, please read the troubleshooting section of the relevant Caffe2 [installation instructions](https://caffe2.ai/docs/getting-started.html) first. In the following, we provide additional troubleshooting tips for Caffe2 and Detectron.

### Caffe2 Operator Profiling

Caffe2 comes with performance [`profiling`](https://github.com/caffe2/caffe2/tree/master/caffe2/contrib/prof)
support which you may find useful for benchmarking or debugging your operators
(see [`BatchPermutationOp test`](tests/test_batch_permutation_op.py) for example usage).
Profiling support is not built by default and you can enable it by setting
the `-DUSE_PROF=ON` flag when running Caffe2 CMake.

### CMake Cannot Find CUDA and cuDNN

Sometimes CMake has trouble with finding CUDA and cuDNN dirs on your machine.

When building Caffe2, you can point CMake to CUDA and cuDNN dirs by running:

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
```

Similarly, when building custom Detectron operators you can use:

```
cd $DETECTRON/lib
mkdir -p build && cd build
cmake .. \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
make
```

Note that you can use the same commands to get CMake to use specific versions of CUDA and cuDNN out of possibly multiple versions installed on your machine.

### Protobuf Errors

Caffe2 uses protobuf as its serialization format and requires version `3.2.0` or newer.
If your protobuf version is older, you can build protobuf from Caffe2 protobuf submodule and use that version instead.

To build Caffe2 protobuf submodule:

```
# CAFFE2=/path/to/caffe2
cd $CAFFE2/third_party/protobuf/cmake
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/c2_tp_protobuf \
  -Dprotobuf_BUILD_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-fPIC"
make install
```

To point Caffe2 CMake to the newly built protobuf:

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPROTOBUF_PROTOC_EXECUTABLE=$HOME/c2_tp_protobuf/bin/protoc \
  -DPROTOBUF_INCLUDE_DIR=$HOME/c2_tp_protobuf/include \
  -DPROTOBUF_LIBRARY=$HOME/c2_tp_protobuf/lib64/libprotobuf.a
```

You may also experience problems with protobuf if you have both system and anaconda packages installed.
This could lead to problems as the versions could be mixed at compile time or at runtime.
This issue can also be overcome by following the commands from above.

### Caffe2 Python Binaries

In case you experience issues with CMake being unable to find the required Python paths when
building Caffe2 Python binaries (e.g. in virtualenv), you can try pointing Caffe2 CMake to python
library and include dir by using:

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPYTHON_LIBRARY=$(python2 -c "from distutils import sysconfig; print(sysconfig.get_python_lib())") \
  -DPYTHON_INCLUDE_DIR=$(python2 -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
```

### Caffe2 with NNPACK Build

Detectron does not require Caffe2 built with NNPACK support. If you face NNPACK related issues during Caffe2 installation, you can safely disable NNPACK by setting the `-DUSE_NNPACK=OFF` CMake flag.

### Caffe2 with OpenCV Build

Analogously to the NNPACK case above, you can disable OpenCV by setting the `-DUSE_OPENCV=OFF` CMake flag.

### COCO API Undefined Symbol Error

If you encounter a COCO API import error due to an undefined symbol, as reported [here](https://github.com/cocodataset/cocoapi/issues/35),
make sure that your python versions are not getting mixed. For instance, this issue may arise if you have
[both system and conda numpy installed](https://stackoverflow.com/questions/36190757/numpy-undefined-symbol-pyfpe-jbuf).

### CMake Cannot Find Caffe2

In case you experience issues with CMake being unable to find the Caffe2 package when building custom operators,
make sure you have run `make install` as part of your Caffe2 installation process.

### Conflicting Imports

Python modules with common names could result in import conflicts.
For instance, a `datasets` module is also found in [tensorflow](https://github.com/tensorflow/tensorflow)
and could cause an error (e.g. `ImportError: cannot import name task_evaluation`)
as discussed [here](https://github.com/facebookresearch/Detectron/issues/20).
If you encounter an import error, please make sure that you are not trying to import the module from another project.
