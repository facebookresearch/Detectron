#!/usr/bin/env python

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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import subprocess
import json
from shutil import rmtree
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.utils.keypoints import get_keypoints

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)

def save_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyJSONEncoder, indent=2)

def condense_json_files(
    json_temp_filenames,
    output_directory,
    thresholds=None,
    video_url="",
    output_filename='detections'
):
    json_merged = []

    for f in json_temp_filenames:
        with open(f, "r") as infile:
            json_merged = json_merged + json.load(infile)

    json_final_output = {
        "data": json_merged,
    }

    if thresholds:
        json_final_output["thresholds"] = thresholds

    if video_url:
        json_final_output["originalVideoUrl"] = video_url

    save_json('{d}/{f}.json'.format(d=output_directory, f=output_filename), json_final_output)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
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
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--num_gpus',
        default=8,
        type=int
    )

    parser.add_argument(
        '--video',
        type=str,
    )
    parser.add_argument(
        '--limit',
        type=int,
    )

    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    if args.video:
        cap = cv2.VideoCapture(args.video)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights, gpu_id=0)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if cap:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        zfill_amount = len(str(frame_count))
        if args.limit:
            frame_count = min(args.limit, frame_count)
        im_list = list(
            map(
                lambda i: str(i).zfill(zfill_amount),
                list(
                    range(frame_count)
                )
            )
        )
    elif os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    json_file_names = []

    json_dir = '{}/json'.format(args.output_dir)

    if os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.isdir(json_dir):
        rmtree(json_dir)

    if not os.path.isdir(json_dir):
        os.mkdir(json_dir)

    frames_dir = '{}/frames'.format(args.output_dir)

    if os.path.isdir(frames_dir):
        rmtree(frames_dir)

    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '/frames/{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )

        out_name_json = '{}/json/{}.json'.format(args.output_dir, im_name)

        json_file_names.append(out_name_json)

        logger.info('Processing {} -> {}'.format(im_name, out_name))
        if cap:
            retval, im = cap.read()
            if retval is False or im is None:
                break
        else:
            im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box
        )

        boxes, _, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)

        keypoints_labels, _ = get_keypoints()

        detection_dict_list = []

        for k in range(len(boxes)):
            # I think its possible for a box and class to be present without
            # keypoints if something other then a human is detected.
            keypoint = keypoints[k]
            keypoints_dict = {}
            for label_i in range(len(keypoints_labels)):
                label = keypoints_labels[label_i]
                keypoints_dict[label] = {
                    'x': int(keypoint[0, label_i]),
                    'y': int(keypoint[1, label_i]),
                    'logit': keypoint[2, label_i],
                    'prob': keypoint[3, label_i],
                    'label': label,
                    'label_index': label_i
                }

            detection_dict = {
                'indices': { 'frame': i, 'detection': k },
                'box': {
                    'xmin': int(boxes[k][0]),
                    'ymin': int(boxes[k][1]),
                    'xmax': int(boxes[k][2]),
                    'ymax': int(boxes[k][3]),
                    'score': boxes[k][4]
                },
                'class': dummy_coco_dataset.classes[int(classes[k])],
                'keypoints': keypoints_dict,
            }

            detection_dict_list.append(detection_dict)

        save_json(
            filename=out_name_json,
            data={
                'detections': detection_dict_list
            }
        )

    condense_json_files(
        json_file_names,
        args.output_dir,
        thresholds={
            'box': args.thresh,
            'keypoints': args.kp_thresh
        }
    )

    # rmtree('{}/json'.format(args.output_dir))

    if cap:
        command = 'ffmpeg -y -framerate {fps} -i {images_dir}/%0{zf}d.jpg {d}/{vn}.mp4'.format(
            d=args.output_dir,
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            images_dir=args.output_dir,
            vn='video', zf=zfill_amount)
        proc = subprocess.Popen(command, shell=True)
        proc.wait()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
