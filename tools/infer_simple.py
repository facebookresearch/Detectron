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

#ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from PIL import Image as PL

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
        default='configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
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
        default='jpg',
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
    #here, i have to edit this to read images directly from the published ros topic
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

class get_image:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)

    def callback(self,data):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        args.im_or_folder = cv_image




        # args.im_or_folder = data

def talker(i):
    image_publisher = rospy.Publisher('detectron_output', Image, queue_size=10)
    brdg = CvBridge()
    image_publisher.publish(brdg.cv2_to_imgmsg(i, "bgr8"))



def main(args):
    #ros stuff
    rospy.init_node('get_image', anonymous=True)
    # rospy.init_node('talker', anonymous=True)


    args.im_or_folder = get_image()


    while not rospy.is_shutdown():

        logger = logging.getLogger(__name__)

        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        model = infer_engine.initialize_model_from_cfg(args.weights)
        dummy_coco_dataset = dummy_datasets.get_coco_dataset()
        #
        # if os.path.isdir(args.im_or_folder):
        #     im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
        # else
        im_list = [args.im_or_folder]

        im_name = "test"
        for i in enumerate(im_list):
            out_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
            )
            logger.info('Processing {} -> {}'.format(im_name, out_name))
            im = args.im_or_folder

            #so im == image from webcam, but the problem is that it only takes a single picture...ignore this for now
            # im = cv2.imread(im_name)
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, im, None, timers=timers
                )
            #last point the engine runs for
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            if i == 0:
                logger.info(
                    ' \ Note: inference on the first image will be slower than the '
                    'rest (caches and auto-tuning need to warm up)'
                )
            cv2.imshow("Raw Image", im)
            fig = vis_utils.vis_one_image(
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


            for i in range(1000):
                # update data
                # redraw the canvas
                fig.canvas.draw()

                # convert canvas to image
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # img is rgb, convert to opencv's default bgr
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # display image with opencv or any operation you like
                # cv2.imshow("plot", img)
                # k = cv2.waitKey(33) & 0xFF
                # if k == 27:
                #     break
        #     fig.canvas.draw()
        #
        #     w, h = fig.canvas.get_width_height()
        #     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        #     buf.shape = (w, h, 4)
        #
        #     buf = np.roll(buf, 3, axis=2)
        #
        #     im = PL.frombytes("RGBA", (w, h), buf.tostring())
        # # im.show()
        #
        #     open_cv_image = np.array(im)
        #     cv2.namedWindow('detected',cv2.WINDOW_NORMAL)
        #     # cv2.resizeWindow('detected', (600,600))
            cv2.imshow("Detected Image..Press any key to repeat the process and publish image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # talker(open_cv_image)
            image_publisher = rospy.Publisher('detectron_output', Image, queue_size=10)
            brdg = CvBridge()
            image_publisher.publish(brdg.cv2_to_imgmsg(img, "bgr8"))

            try:

                # while(1):
                print('Finished..Hold Ctrl+C to end')
                    # args = parse_args()
                main(args)
            except KeyboardInterrupt:
                print("shutting down")
                cv2.destroyAllWindows()




if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
