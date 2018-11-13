
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

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


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

    parser.add_argument(
        'skip_frames',
        type=int
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def smoother(measurements, n_iter=5, last_measurement=None):
    from pykalman import KalmanFilter
    transition_matrix = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0], [0, 0, 1, 0]]

    if last_measurement is None:
        initial_state_mean = [measurements[0, 0], 0, measurements[0, 1], 0]
    else:
        initial_state_mean = [last_measurement[0], 0, last_measurement[1], 0]

    kf1 = KalmanFilter(
        transition_matrices = transition_matrix,
        observation_matrices = observation_matrix,
        initial_state_mean = initial_state_mean
    )

    kf1 = kf1.em(measurements, n_iter=n_iter)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    output = []

    for smoothed_state_mean in smoothed_state_means:
        output.append((smoothed_state_mean[0], smoothed_state_mean[2]))

    return np.asarray(output)

def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = args.num_gpus
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    if args.video:
        cap = cv2.VideoCapture(args.video)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

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

    output_names = []
    grouped_res = []
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        output_names.append(out_name)
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

        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)
        print('boxes', boxes)
        print('segms', segms)
        print('keypoints', keypoints)
        keypoints_labels = [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle'
        ]
        keypoint_dict_list = []
        for keypoint in keypoints:
            keypoint_dict = {}
            for label_i in range(len(keypoints_labels)):
                label = keypoints_labels[label_i]
                keypoint_dict[label] = {
                    'x': int(keypoint[0, label_i]),
                    'y': int(keypoint[1, label_i]),
                    'logit': keypoint[2, label_i],
                    'prob': keypoint[3, label_i],
                }
            keypoint_dict_list.append(keypoint_dict)
        print('keypoint_dict_list: ', keypoint_dict_list)
        grouped_res.append((
            cls_boxes, cls_segms, cls_keyps, im, im_name
        ))


    # import numpy as np
    # from numpy import ma
    # kf = UnscentedKalmanFilter(
    #     lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
    # for k in range(len(grouped_res)):
    #     cls_boxes, cls_segms, cls_keyps, im, im_name = grouped_res[k]

    #     for j in range(len(cls_keyps)):
    #         X = ma.array(cls_keyps[j])
    #         X[1::2] = ma.masked  # hide measurement at time step 1
    #         grouped_res[k][2] = kf.smooth(X)
            # cls_keyps[j] = kf.em(X, n_iter=5).smooth(X)

    for res in grouped_res:
        cls_boxes, cls_segms, cls_keyps, im, im_name = res
        # print('vis_utils: ', vis_utils.vis_one_image(
        #     im[:, :, ::-1],  # BGR -> RGB for visualization
        #     im_name,
        #     args.output_dir,
        #     cls_boxes,
        #     cls_segms,
        #     cls_keyps,
        #     dataset=dummy_coco_dataset,
        #     box_alpha=0.3,
        #     show_class=True,
        #     thresh=args.thresh,
        #     kp_thresh=args.kp_thresh,
        #     ext=args.output_ext,
        #     out_when_no_box=args.out_when_no_box
        # ))
        opencv_image, keypoints_clean = vis_utils.vis_one_image_opencv(
            im,
            cls_boxes,
            segms=cls_segms,
            keypoints=cls_keyps,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            show_box=True,
            dataset=dummy_coco_dataset,
            show_class=False,
        )

        opencv_image_filename = '{}/{}.{}'.format(args.output_dir, im_name, args.output_ext)
        cv2.imwrite(opencv_image_filename, opencv_image)
        print('keypoints_clean: ', keypoints_clean)
        keypoint_x = []
        keypoint_y = []
        grouped_keypoints = []
        for keypoint_index in range(16):
            grouped_keypoints.append(
                list(filter(lambda x: keypoint_index == x['index'], keypoints_clean))
            )

        grouped_smoothed_measurements = []
        from numpy import ma
        print('grouped_keypoints: ', grouped_keypoints)
        for group_keypoint in grouped_keypoints:
            measurements = ma.empty(
                shape=(
                    len(group_keypoint),
                    2
                )
            )
            for ki in range(len(group_keypoint)):
                keypoint = group_keypoint[ki]
                print('keypoint: ', keypoint)
            #     measurements[ki][0] = float(keypoint['y'])
            #     measurements[ki][1] = float(keypoint['x'])
            # smoothed_measurements = smoother(measurements)
            # grouped_smoothed_measurements.append(smoothed_measurements)
        print('grouped_smoothed_measurements: ', grouped_smoothed_measurements)

    if cap:
        import subprocess

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