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

"""Helpful utilities for working with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six import string_types
import contextlib
import subprocess

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import dyndep
from caffe2.python import scope
from caffe2.python import workspace

import detectron.utils.env as envu


def import_contrib_ops():
    """Import contrib ops needed by Detectron."""
    envu.import_nccl_ops()


def import_detectron_ops():
    """Import Detectron ops."""
    detectron_ops_lib = envu.get_detectron_ops_lib()
    dyndep.InitOpsLibrary(detectron_ops_lib)


def import_custom_ops():
    """Import custom ops."""
    custom_ops_lib = envu.get_custom_ops_lib()
    dyndep.InitOpsLibrary(custom_ops_lib)


def SuffixNet(name, net, prefix_len, outputs):
    """Returns a new Net from the given Net (`net`) that includes only the ops
    after removing the first `prefix_len` number of ops. The new Net is thus a
    suffix of `net`. Blobs listed in `outputs` are registered as external output
    blobs.
    """
    outputs = BlobReferenceList(outputs)
    for output in outputs:
        assert net.BlobIsDefined(output)
    new_net = net.Clone(name)

    del new_net.Proto().op[:]
    del new_net.Proto().external_input[:]
    del new_net.Proto().external_output[:]

    # Add suffix ops
    new_net.Proto().op.extend(net.Proto().op[prefix_len:])
    # Add external input blobs
    # Treat any undefined blobs as external inputs
    input_names = [
        i for op in new_net.Proto().op for i in op.input
        if not new_net.BlobIsDefined(i)]
    new_net.Proto().external_input.extend(input_names)
    # Add external output blobs
    output_names = [str(o) for o in outputs]
    new_net.Proto().external_output.extend(output_names)
    return new_net, [new_net.GetBlobRef(o) for o in output_names]


def BlobReferenceList(blob_ref_or_list):
    """Ensure that the argument is returned as a list of BlobReferences."""
    if isinstance(blob_ref_or_list, core.BlobReference):
        return [blob_ref_or_list]
    elif type(blob_ref_or_list) in (list, tuple):
        for b in blob_ref_or_list:
            assert isinstance(b, core.BlobReference)
        return blob_ref_or_list
    else:
        raise TypeError(
            'blob_ref_or_list must be a BlobReference or a list/tuple of '
            'BlobReferences'
        )


def UnscopeName(possibly_scoped_name):
    """Remove any name scoping from a (possibly) scoped name. For example,
    convert the name 'gpu_0/foo' to 'foo'."""
    assert isinstance(possibly_scoped_name, string_types)
    return possibly_scoped_name[
        possibly_scoped_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


@contextlib.contextmanager
def NamedCudaScope(gpu_id):
    """Creates a GPU name scope and CUDA device scope. This function is provided
    to reduce `with ...` nesting levels."""
    with GpuNameScope(gpu_id):
        with CudaScope(gpu_id):
            yield


@contextlib.contextmanager
def GpuNameScope(gpu_id):
    """Create a name scope for GPU device `gpu_id`."""
    with core.NameScope('gpu_{:d}'.format(gpu_id)):
        yield


@contextlib.contextmanager
def CudaScope(gpu_id):
    """Create a CUDA device scope for GPU device `gpu_id`."""
    gpu_dev = CudaDevice(gpu_id)
    with core.DeviceScope(gpu_dev):
        yield


@contextlib.contextmanager
def CpuScope():
    """Create a CPU device scope."""
    cpu_dev = core.DeviceOption(caffe2_pb2.CPU)
    with core.DeviceScope(cpu_dev):
        yield


def CudaDevice(gpu_id):
    """Create a Cuda device."""
    return core.DeviceOption(caffe2_pb2.CUDA, gpu_id)


def gauss_fill(std):
    """Gaussian fill helper to reduce verbosity."""
    return ('GaussianFill', {'std': std})


def const_fill(value):
    """Constant fill helper to reduce verbosity."""
    return ('ConstantFill', {'value': value})


def get_nvidia_info():
    return (
        get_nvidia_smi_output(),
        workspace.GetCUDAVersion(),
        workspace.GetCuDNNVersion(),
    )


def get_nvidia_smi_output():
    try:
        info = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        info = info.decode("utf8")
    except Exception as e:
        info = "Executing nvidia-smi failed: " + str(e)
    return info.strip()
