# Convert a detection model trained for COCO into a model that can be fine-tuned
# on cityscapes
#
# cityscapes_to_coco

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import argparse
import os
import sys
import numpy as np

import datasets.cityscapes.coco_to_cityscapes_id as cs

NUM_CS_CLS = 9
NUM_COCO_CLS = 81


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a COCO pre-trained model for use with Cityscapes')
    parser.add_argument(
        '--coco_model', dest='coco_model_file_name',
        help='Pretrained network weights file path',
        default=None, type=str)
    parser.add_argument(
        '--convert_func', dest='convert_func',
        help='Blob conversion function',
        default='cityscapes_to_coco', type=str)
    parser.add_argument(
        '--output', dest='out_file_name',
        help='Output file path',
        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def convert_coco_blobs_to_cityscape_blobs(model_dict):
    for k, v in model_dict['blobs'].items():
        if v.shape[0] == NUM_COCO_CLS or v.shape[0] == 4 * NUM_COCO_CLS:
            coco_blob = model_dict['blobs'][k]
            print(
                'Converting COCO blob {} with shape {}'.
                format(k, coco_blob.shape)
            )
            cs_blob = convert_coco_blob_to_cityscapes_blob(
                coco_blob, args.convert_func
            )
            print(' -> converted shape {}'.format(cs_blob.shape))
            model_dict['blobs'][k] = cs_blob


def convert_coco_blob_to_cityscapes_blob(coco_blob, convert_func):
    # coco blob (81, ...) or (81*4, ...)
    coco_shape = coco_blob.shape
    leading_factor = int(coco_shape[0] / NUM_COCO_CLS)
    tail_shape = list(coco_shape[1:])
    assert leading_factor == 1 or leading_factor == 4

    # Reshape in [num_classes, ...] form for easier manipulations
    coco_blob = coco_blob.reshape([NUM_COCO_CLS, -1] + tail_shape)
    # Default initialization uses Gaussian with mean and std to match the
    # existing parameters
    std = coco_blob.std()
    mean = coco_blob.mean()
    cs_shape = [NUM_CS_CLS] + list(coco_blob.shape[1:])
    cs_blob = (np.random.randn(*cs_shape) * std + mean).astype(np.float32)

    # Replace random parameters with COCO parameters if class mapping exists
    for i in range(NUM_CS_CLS):
        coco_cls_id = getattr(cs, convert_func)(i)
        if coco_cls_id >= 0:  # otherwise ignore (rand init)
            cs_blob[i] = coco_blob[coco_cls_id]

    cs_shape = [NUM_CS_CLS * leading_factor] + tail_shape
    return cs_blob.reshape(cs_shape)


def remove_momentum(model_dict):
    for k in model_dict['blobs'].keys():
        if k.endswith('_momentum'):
            del model_dict['blobs'][k]


def load_and_convert_coco_model(args):
    with open(args.coco_model_file_name, 'r') as f:
        model_dict = pickle.load(f)
    remove_momentum(model_dict)
    convert_coco_blobs_to_cityscape_blobs(model_dict)
    return model_dict


if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.coco_model_file_name), \
        'Weights file does not exist'
    weights = load_and_convert_coco_model(args)

    with open(args.out_file_name, 'w') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote blobs to {}:'.format(args.out_file_name))
    print(sorted(weights['blobs'].keys()))
