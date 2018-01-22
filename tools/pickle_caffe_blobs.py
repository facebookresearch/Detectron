#!/usr/bin/env python2

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

"""Script for converting Caffe (<= 1.0) models into the the simple state dict
format used by Detectron. For example, this script can convert the orignal
ResNet models released by MSRA.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import os
import sys

from caffe.proto import caffe_pb2
from caffe2.proto import caffe2_pb2
from caffe2.python import caffe_translator
from caffe2.python import utils
from google.protobuf import text_format


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump weights from a Caffe model'
    )
    parser.add_argument(
        '--prototxt',
        dest='prototxt_file_name',
        help='Network definition prototxt file path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--caffemodel',
        dest='caffemodel_file_name',
        help='Pretrained network weights file path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output',
        dest='out_file_name',
        help='Output file path',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def normalize_resnet_name(name):
    if name.find('res') == 0 and name.find('res_') == -1:
        # E.g.,
        #  res4b11_branch2c -> res4_11_branch2c
        #  res2a_branch1 -> res2_0_branch1
        chunk = name[len('res'):name.find('_')]
        name = (
            'res' + chunk[0] + '_' + str(
                int(chunk[2:]) if len(chunk) > 2  # e.g., "b1" -> 1
                else ord(chunk[1]) - ord('a')
            ) +  # e.g., "a" -> 0
            name[name.find('_'):]
        )
    return name


def pickle_weights(out_file_name, weights):
    blobs = {
        normalize_resnet_name(blob.name): utils.Caffe2TensorToNumpyArray(blob)
        for blob in weights.protos
    }
    with open(out_file_name, 'w') as f:
        pickle.dump(blobs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote blobs:')
    print(sorted(blobs.keys()))


def add_missing_biases(caffenet_weights):
    for layer in caffenet_weights.layer:
        if layer.type == 'Convolution' and len(layer.blobs) == 1:
            num_filters = layer.blobs[0].shape.dim[0]
            bias_blob = caffe_pb2.BlobProto()
            bias_blob.data.extend(np.zeros(num_filters))
            bias_blob.num, bias_blob.channels, bias_blob.height = 1, 1, 1
            bias_blob.width = num_filters
            layer.blobs.extend([bias_blob])


def remove_spatial_bn_layers(caffenet, caffenet_weights):
    # Layer types associated with spatial batch norm
    remove_types = ['BatchNorm', 'Scale']

    def _remove_layers(net):
        for i in reversed(range(len(net.layer))):
            if net.layer[i].type in remove_types:
                net.layer.pop(i)

    # First remove layers from caffenet proto
    _remove_layers(caffenet)
    # We'll return these so we can save the batch norm parameters
    bn_layers = [
        layer for layer in caffenet_weights.layer if layer.type in remove_types
    ]
    _remove_layers(caffenet_weights)

    def _create_tensor(arr, shape, name):
        t = caffe2_pb2.TensorProto()
        t.name = name
        t.data_type = caffe2_pb2.TensorProto.FLOAT
        t.dims.extend(shape.dim)
        t.float_data.extend(arr)
        assert len(t.float_data) == np.prod(t.dims), 'Data size, shape mismatch'
        return t

    bn_tensors = []
    for (bn, scl) in zip(bn_layers[0::2], bn_layers[1::2]):
        assert bn.name[len('bn'):] == scl.name[len('scale'):], 'Pair mismatch'
        blob_out = 'res' + bn.name[len('bn'):] + '_bn'
        bn_mean = np.asarray(bn.blobs[0].data)
        bn_var = np.asarray(bn.blobs[1].data)
        scale = np.asarray(scl.blobs[0].data)
        bias = np.asarray(scl.blobs[1].data)
        std = np.sqrt(bn_var + 1e-5)
        new_scale = scale / std
        new_bias = bias - bn_mean * scale / std
        new_scale_tensor = _create_tensor(
            new_scale, bn.blobs[0].shape, blob_out + '_s'
        )
        new_bias_tensor = _create_tensor(
            new_bias, bn.blobs[0].shape, blob_out + '_b'
        )
        bn_tensors.extend([new_scale_tensor, new_bias_tensor])
    return bn_tensors


def remove_layers_without_parameters(caffenet, caffenet_weights):
    for i in reversed(range(len(caffenet_weights.layer))):
        if len(caffenet_weights.layer[i].blobs) == 0:
            # Search for the corresponding layer in caffenet and remove it
            name = caffenet_weights.layer[i].name
            found = False
            for j in range(len(caffenet.layer)):
                if caffenet.layer[j].name == name:
                    caffenet.layer.pop(j)
                    found = True
                    break
            if not found and name[-len('_split'):] != '_split':
                print('Warning: layer {} not found in caffenet'.format(name))
            caffenet_weights.layer.pop(i)


def normalize_shape(caffenet_weights):
    for layer in caffenet_weights.layer:
        for blob in layer.blobs:
            shape = (blob.num, blob.channels, blob.height, blob.width)
            if len(blob.data) != np.prod(shape):
                shape = tuple(blob.shape.dim)
                if len(shape) == 1:
                    # Handle biases
                    shape = (1, 1, 1, shape[0])
                if len(shape) == 2:
                    # Handle InnerProduct layers
                    shape = (1, 1, shape[0], shape[1])
                assert len(shape) == 4
                blob.num, blob.channels, blob.height, blob.width = shape


def load_and_convert_caffe_model(prototxt_file_name, caffemodel_file_name):
    caffenet = caffe_pb2.NetParameter()
    caffenet_weights = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file_name).read(), caffenet)
    caffenet_weights.ParseFromString(open(caffemodel_file_name).read())
    # C2 conv layers current require biases, but they are optional in C1
    # Add zeros as biases is they are missing
    add_missing_biases(caffenet_weights)
    # We only care about getting parameters, so remove layers w/o parameters
    remove_layers_without_parameters(caffenet, caffenet_weights)
    # BatchNorm is not implemented in the translator *and* we need to fold Scale
    # layers into the new C2 SpatialBN op, hence we remove the batch norm layers
    # and apply custom translations code
    bn_weights = remove_spatial_bn_layers(caffenet, caffenet_weights)
    # Set num, channel, height and width for blobs that use shape.dim instead
    normalize_shape(caffenet_weights)
    # Translate the rest of the model
    net, pretrained_weights = caffe_translator.TranslateModel(
        caffenet, caffenet_weights
    )
    pretrained_weights.protos.extend(bn_weights)
    return net, pretrained_weights


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.prototxt_file_name), \
        'Prototxt file does not exist'
    assert os.path.exists(args.caffemodel_file_name), \
        'Weights file does not exist'
    net, weights = load_and_convert_caffe_model(
        args.prototxt_file_name, args.caffemodel_file_name
    )
    pickle_weights(args.out_file_name, weights)
