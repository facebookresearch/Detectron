from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import sys

from caffe2.python import net_drawer

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Network Visualization')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_model(cfg_file, weights_file):
    merge_cfg_from_file(cfg_file)
    cfg.TRAIN.WEIGHTS = ''  # NOTE: do not download pretrained model weights
    cfg.TEST.WEIGHTS = weights_file
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS)
    return model


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args.cfg, args.weights)

    g = net_drawer.GetPydotGraph(model, rankdir="TB")
    g.write_dot(model.Proto().name + '.dot')

