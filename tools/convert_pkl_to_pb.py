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

"""Script to convert the model (.yaml and .pkl) trained by train_net to a
standard Caffe2 model in pb format (model.pb and model_init.pb). The converted
model is good for production usage, as it could run independently and efficiently
on CPU, GPU and mobile without depending on the detectron codebase.

Please see Caffe2 tutorial (
https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html) for loading
the converted model, and run_model_pb() for running the model for inference.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os
import pprint
import sys

import caffe2.python.utils as putils
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.modeling import generate_anchors
from detectron.utils.logging import setup_logging
from detectron.utils.model_convert_utils import convert_op_in_proto
from detectron.utils.model_convert_utils import op_filter
import detectron.core.test_engine as test_engine
import detectron.core.test as test
import detectron.utils.c2 as c2_utils
import detectron.utils.model_convert_utils as mutils
import detectron.utils.vis as vis_utils
import detectron.utils.blob as blob_utils
import detectron.utils.keypoints as keypoint_utils
import pycocotools.mask as mask_utils

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = setup_logging(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a trained network to pb format'
    )
    parser.add_argument(
        '--cfg', dest='cfg_file', help='optional config file', default=None,
        type=str)
    parser.add_argument(
        '--net_name', dest='net_name', help='optional name for the net',
        default="detectron", type=str)
    parser.add_argument(
        '--out_dir', dest='out_dir', help='output dir', default=None,
        type=str)
    parser.add_argument(
        '--test_img', dest='test_img',
        help='optional test image, used to verify the model conversion',
        default=None,
        type=str)
    parser.add_argument(
        '--fuse_af', dest='fuse_af', help='1 to fuse_af',
        default=1,
        type=int)
    parser.add_argument(
        '--device', dest='device',
        help='Device to run the model on',
        choices=['cpu', 'gpu'],
        default='cpu',
        type=str)
    parser.add_argument(
        '--net_execution_type', dest='net_execution_type',
        help='caffe2 net execution type',
        choices=['simple', 'dag'],
        default='simple',
        type=str)
    parser.add_argument(
        '--use_nnpack', dest='use_nnpack',
        help='Use nnpack for conv',
        default=1,
        type=int)
    parser.add_argument(
        'opts', help='See detectron/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    ret = parser.parse_args()
    ret.out_dir = os.path.abspath(ret.out_dir)
    if ret.device == 'gpu' and ret.use_nnpack:
        logger.warn('Should not use mobile engine for gpu model.')
        ret.use_nnpack = 0

    return ret


def unscope_name(name):
    return c2_utils.UnscopeName(name)


def reset_names(names):
    for i in range(0, len(names)):
        names[i] = unscope_name(names[i])


def convert_gen_proposals(
    op, blobs,
    rpn_pre_nms_topN,
    rpn_post_nms_topN,
    rpn_nms_thres,
    rpn_min_size,
):
    print('Converting GenerateProposals Python -> C++:\n{}'.format(op))
    assert op.name.startswith("GenerateProposalsOp"), "Not valid GenerateProposalsOp"

    spatial_scale = mutils.get_op_arg_valf(op, "spatial_scale", None)
    assert spatial_scale is not None

    inputs = [x for x in op.input]
    anchor_name = "anchor"
    inputs.append(anchor_name)
    blobs[anchor_name] = get_anchors(spatial_scale)
    print('anchors {}'.format(blobs[anchor_name]))

    ret = core.CreateOperator(
        "GenerateProposals",
        inputs,
        list(op.output),
        spatial_scale=spatial_scale,
        pre_nms_topN=rpn_pre_nms_topN,
        post_nms_topN=rpn_post_nms_topN,
        nms_thres=rpn_nms_thres,
        min_size=rpn_min_size,
        correct_transform_coords=True,
    )

    return ret, anchor_name


def get_anchors(spatial_scale):
    anchors = generate_anchors.generate_anchors(
        stride=1. / spatial_scale,
        sizes=cfg.RPN.SIZES,
        aspect_ratios=cfg.RPN.ASPECT_RATIOS).astype(np.float32)
    return anchors


def reset_blob_names(blobs):
    ret = {unscope_name(x): blobs[x] for x in blobs}
    blobs.clear()
    blobs.update(ret)


def convert_net(args, net, blobs):

    @op_filter()
    def convert_op_name(op):
        if args.device != 'gpu':
            if op.engine != 'DEPTHWISE_3x3':
                op.engine = ''
            op.device_option.CopyFrom(caffe2_pb2.DeviceOption())
        reset_names(op.input)
        reset_names(op.output)
        return [op]

    @op_filter(type="Python", inputs=['rpn_cls_probs', 'rpn_bbox_pred', 'im_info'])
    def convert_gen_proposal(op_in):
        gen_proposals_op, ext_input = convert_gen_proposals(
            op_in, blobs,
            rpn_min_size=float(cfg.TEST.RPN_MIN_SIZE),
            rpn_post_nms_topN=cfg.TEST.RPN_POST_NMS_TOP_N,
            rpn_pre_nms_topN=cfg.TEST.RPN_PRE_NMS_TOP_N,
            rpn_nms_thres=cfg.TEST.RPN_NMS_THRESH,
        )
        net.external_input.extend([ext_input])
        return [gen_proposals_op]

    @op_filter(input_has='rois')
    def convert_rpn_rois(op):
        for j in range(0, len(op.input)):
            if op.input[j] == 'rois':
                print('Converting op {} input name: rois -> rpn_rois:\n{}'.format(
                    op.type, op))
                op.input[j] = 'rpn_rois'
        return [op]

    @op_filter(type_in=['StopGradient', 'Alias'])
    def convert_remove_op(op):
        print('Removing op {}:\n{}'.format(op.type, op))
        return []

    convert_op_in_proto(net, convert_op_name)
    convert_op_in_proto(net, [
        convert_gen_proposal, convert_rpn_rois, convert_remove_op
    ])

    reset_names(net.external_input)
    reset_names(net.external_output)

    reset_blob_names(blobs)


def add_bbox_ops(args, net, blobs):
    new_ops = []
    new_external_outputs = []

    # Operators for bboxes
    op_box = core.CreateOperator(
        "BBoxTransform",
        ['rpn_rois', 'bbox_pred', 'im_info'],
        ['pred_bbox'],
        weights=cfg.MODEL.BBOX_REG_WEIGHTS,
        apply_scale=False,
        correct_transform_coords=True,
    )
    new_ops.extend([op_box])

    blob_prob = 'cls_prob'
    blob_box = 'pred_bbox'
    op_nms = core.CreateOperator(
        "BoxWithNMSLimit",
        [blob_prob, blob_box],
        ['score_nms', 'bbox_nms', 'class_nms'],
        arg=[
            putils.MakeArgument("score_thresh", cfg.TEST.SCORE_THRESH),
            putils.MakeArgument("nms", cfg.TEST.NMS),
            putils.MakeArgument("detections_per_im", cfg.TEST.DETECTIONS_PER_IM),
            putils.MakeArgument("soft_nms_enabled", cfg.TEST.SOFT_NMS.ENABLED),
            putils.MakeArgument("soft_nms_method", cfg.TEST.SOFT_NMS.METHOD),
            putils.MakeArgument("soft_nms_sigma", cfg.TEST.SOFT_NMS.SIGMA),
        ]
    )
    new_ops.extend([op_nms])
    new_external_outputs.extend(['score_nms', 'bbox_nms', 'class_nms'])

    net.Proto().op.extend(new_ops)
    net.Proto().external_output.extend(new_external_outputs)


def convert_model_gpu(args, net, init_net):
    assert args.device == 'gpu'

    ret_net = copy.deepcopy(net)
    ret_init_net = copy.deepcopy(init_net)

    cdo_cuda = mutils.get_device_option_cuda()
    cdo_cpu = mutils.get_device_option_cpu()

    CPU_OPS = [
        ["GenerateProposals", None],
        ["BBoxTransform", None],
        ["BoxWithNMSLimit", None],
    ]
    CPU_BLOBS = ["im_info", "anchor"]

    @op_filter()
    def convert_op_gpu(op):
        for x in CPU_OPS:
            if mutils.filter_op(op, type=x[0], inputs=x[1]):
                return None
        op.device_option.CopyFrom(cdo_cuda)
        return [op]

    @op_filter()
    def convert_init_op_gpu(op):
        if op.output[0] in CPU_BLOBS:
            op.device_option.CopyFrom(cdo_cpu)
        else:
            op.device_option.CopyFrom(cdo_cuda)
        return [op]

    convert_op_in_proto(ret_init_net.Proto(), convert_init_op_gpu)
    convert_op_in_proto(ret_net.Proto(), convert_op_gpu)

    ret = core.InjectDeviceCopiesAmongNets([ret_init_net, ret_net])

    return [ret[0][1], ret[0][0]]


def gen_init_net(net, blobs, empty_blobs):
    blobs = copy.deepcopy(blobs)
    for x in empty_blobs:
        blobs[x] = np.array([], dtype=np.float32)
    init_net = mutils.gen_init_net_from_blobs(
        blobs, net.external_inputs)
    init_net = core.Net(init_net)
    return init_net


def _save_image_graphs(args, all_net, all_init_net):
    print('Saving model graph...')
    mutils.save_graph(
        all_net.Proto(), os.path.join(args.out_dir, all_net.Proto().name + '.png'),
        op_only=False)
    print('Model def image saved to {}.'.format(args.out_dir))


def _save_models(all_net, all_init_net, args):
    print('Writing converted model to {}...'.format(args.out_dir))
    fname = all_net.Proto().name

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.out_dir, fname + '.pb'), 'wb') as f:
        f.write(all_net.Proto().SerializeToString())
    with open(os.path.join(args.out_dir, fname + '.pbtxt'), 'w') as f:
        f.write(str(all_net.Proto()))
    with open(os.path.join(args.out_dir, fname + '_init.pb'), 'wb') as f:
        f.write(all_init_net.Proto().SerializeToString())

    _save_image_graphs(args, all_net, all_init_net)


def load_model(args):
    model = test_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS)
    blobs = mutils.get_ws_blobs()

    return model, blobs


def _get_result_blobs(check_blobs):
    ret = {}
    for x in check_blobs:
        sn = core.ScopedName(x)
        if workspace.HasBlob(sn):
            ret[x] = workspace.FetchBlob(sn)
        else:
            ret[x] = None

    return ret


def _sort_results(boxes, segms, keypoints, classes):
    indices = np.argsort(boxes[:, -1])[::-1]
    if boxes is not None:
        boxes = boxes[indices, :]
    if segms is not None:
        segms = [segms[x] for x in indices]
    if keypoints is not None:
        keypoints = [keypoints[x] for x in indices]
    if classes is not None:
        if isinstance(classes, list):
            classes = [classes[x] for x in indices]
        else:
            classes = classes[indices]

    return boxes, segms, keypoints, classes


def run_model_cfg(args, im, check_blobs):
    workspace.ResetWorkspace()
    model, _ = load_model(args)
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = test_engine.im_detect_all(
            model, im, None, None,
        )
    boxes, segms, keypoints, classids = vis_utils.convert_from_cls_format(
        cls_boxes, cls_segms, cls_keyps)

    segms = mask_utils.decode(segms) if segms else None

    # sort the results based on score for comparision
    boxes, segms, keypoints, classids = _sort_results(
        boxes, segms, keypoints, classids)

    # write final results back to workspace
    def _ornone(res):
        return np.array(res) if res is not None else np.array([], dtype=np.float32)
    with c2_utils.NamedCudaScope(0):
        workspace.FeedBlob(core.ScopedName('result_boxes'), _ornone(boxes))
        workspace.FeedBlob(core.ScopedName('result_segms'), _ornone(segms))
        workspace.FeedBlob(core.ScopedName('result_keypoints'), _ornone(keypoints))
        workspace.FeedBlob(core.ScopedName('result_classids'), _ornone(classids))

    # get result blobs
    with c2_utils.NamedCudaScope(0):
        ret = _get_result_blobs(check_blobs)

    print('result_boxes', _ornone(boxes))
    print('result_segms', _ornone(segms))
    print('result_keypoints', _ornone(keypoints))
    print('result_classids', _ornone(classids))
    return ret


def _prepare_blobs(
    im,
    pixel_means,
    target_size,
    max_size,
):
    ''' Reference: blob.prep_im_for_blob() '''

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    blob = np.zeros([1, im.shape[0], im.shape[1], 3], dtype=np.float32)
    blob[0, :, :, :] = im
    channel_swap = (0, 3, 1, 2)  # swap channel to (k, c, h, w)
    blob = blob.transpose(channel_swap)

    blobs = {}
    blobs['data'] = blob
    blobs['im_info'] = np.array(
        [[blob.shape[2], blob.shape[3], im_scale]],
        dtype=np.float32
    )
    return blobs


def run_model_pb(args, models_pb, im, check_blobs):
    workspace.ResetWorkspace()
    net, init_net = models_pb['net']
    workspace.RunNetOnce(init_net)
    mutils.create_input_blobs_for_net(net.Proto())
    workspace.CreateNet(net)

    input_blobs = _prepare_blobs(
        im,
        cfg.PIXEL_MEANS,
        cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
    )
    gpu_blobs = []
    if args.device == 'gpu':
        gpu_blobs = ['data']
    for k, v in input_blobs.items():
        workspace.FeedBlob(
            core.ScopedName(k),
            v,
            mutils.get_device_option_cuda() if k in gpu_blobs else
            mutils.get_device_option_cpu()
        )

    try:
        workspace.RunNet(net)
        scores = workspace.FetchBlob(core.ScopedName('score_nms'))
        classids = workspace.FetchBlob(core.ScopedName('class_nms'))
        boxes = workspace.FetchBlob(core.ScopedName('bbox_nms'))
    except Exception as e:
        print('Running pb model failed.\n{}'.format(e))
        R = 0
        scores = np.zeros((R,), dtype=np.float32)
        boxes = np.zeros((R, 4), dtype=np.float32)
        classids = np.zeros((R,), dtype=np.float32)

    cls_keyps, cls_segms = None, None

    if 'keypoint_net' in models_pb:
        keypoint_net, init_keypoint_net = models_pb['keypoint_net']
        workspace.RunNetOnce(init_keypoint_net)
        mutils.create_input_blobs_for_net(keypoint_net.Proto())
        keypoint_net.Proto().external_input.extend(['rpn_rois', 'bbox_pred', 'im_info', 'cls_prob'])
        workspace.CreateNet(keypoint_net)

        im_scale = input_blobs['im_info'][0][2]
        input_blobs = {'keypoint_rois': test._get_rois_blob(boxes, im_scale)}

        # Add multi-level rois for FPN
        if cfg.FPN.MULTILEVEL_ROIS:
            test._add_multilevel_rois_for_test(input_blobs, 'keypoint_rois')

        gpu_blobs = []
        if args.device == 'gpu':
            gpu_blobs = ['data']
        for k, v in list(input_blobs.items()):
            workspace.FeedBlob(
                core.ScopedName(k),
                v,
                mutils.get_device_option_cuda() if k in gpu_blobs else
                mutils.get_device_option_cpu()
            )

        try:
            workspace.RunNet(keypoint_net)
            pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()
            # In case of 1
            if pred_heatmaps.ndim == 3:
                pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)
        except Exception as e:
            print('Running pb model failed.\n{}'.format(e))
            R, M = 0, cfg.KRCNN.HEATMAP_SIZE
            pred_heatmaps = np.zeros((R, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)

        xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, boxes)
        cls_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
        cls_keyps[1] = [xy_preds[i] for i in range(xy_preds.shape[0])]

    if 'mask_net' in models_pb:
        mask_net, init_mask_net = models_pb['mask_net']
        workspace.RunNetOnce(init_mask_net)
        mutils.create_input_blobs_for_net(mask_net.Proto())
        mask_net.Proto().external_input.extend(['rpn_rois', 'bbox_pred', 'im_info', 'cls_prob'])
        workspace.CreateNet(mask_net)

        im_scale = input_blobs['im_info'][0][2]
        input_blobs = {'mask_rois': test._get_rois_blob(boxes, im_scale)}

        # Add multi-level rois for FPN
        if cfg.FPN.MULTILEVEL_ROIS:
            test._add_multilevel_rois_for_test(input_blobs, 'mask_rois')

        gpu_blobs = []
        if args.device == 'gpu':
            gpu_blobs = ['data']
        for k, v in list(input_blobs.items()):
            workspace.FeedBlob(
                core.ScopedName(k),
                v,
                mutils.get_device_option_cuda() if k in gpu_blobs else
                mutils.get_device_option_cpu()
            )
        M = cfg.MRCNN.RESOLUTION
        try:
            workspace.RunNet(mask_net)
            # Fetch masks
            pred_masks = workspace.FetchBlob(core.ScopedName('mask_fcn_probs')).squeeze()
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
            else:
                pred_masks = pred_masks.reshape([-1, 1, M, M])
        except Exception as e:
            print('Running pb model failed.\n{}'.format(e))
            R = 0
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                pred_masks = np.zeros((R, cfg.MODEL.NUM_CLASSES, M, M), dtype=np.float32)
            else:
                pred_masks = np.zeros((R, 1, M, M), dtype=np.float32)

        cls_boxes = [np.empty(list(classids).count(i)) for i in range(cfg.MODEL.NUM_CLASSES)]
        cls_segms = test.segm_results(cls_boxes, pred_masks, boxes, im.shape[0], im.shape[1])

    boxes = np.column_stack((boxes, scores))

    _, segms, keypoints, _ = vis_utils.convert_from_cls_format([], cls_segms, cls_keyps)
    segms = mask_utils.decode(segms) if segms else None

    # sort the results based on score for comparision
    boxes, segms, keypoints, classids = _sort_results(
        boxes, segms, keypoints, classids)

    # write final result back to workspace
    def _ornone(res):
        return np.array(res) if res is not None else np.array([], dtype=np.float32)
    workspace.FeedBlob(core.ScopedName('result_boxes'), _ornone(boxes))
    workspace.FeedBlob(core.ScopedName('result_classids'), _ornone(classids))
    workspace.FeedBlob(core.ScopedName('result_segms'), _ornone(segms))
    workspace.FeedBlob(core.ScopedName('result_keypoints'), _ornone(keypoints))

    ret = _get_result_blobs(check_blobs)

    print('result_boxes', _ornone(boxes))
    print('result_segms', _ornone(segms))
    print('result_keypoints', _ornone(keypoints))
    print('result_classids', _ornone(classids))
    return ret


def verify_model(args, models_pb, test_img_file):
    check_blobs = ['result_boxes', 'result_classids']

    if cfg.MODEL.MASK_ON:
        check_blobs.append('result_segms')

    if cfg.MODEL.KEYPOINTS_ON:
        check_blobs.append('result_keypoints')

    print('Loading test file {}...'.format(test_img_file))
    test_img = cv2.imread(test_img_file)
    assert test_img is not None

    def _run_cfg_func(im, blobs):
        return run_model_cfg(args, im, check_blobs)

    def _run_pb_func(im, blobs):
        return run_model_pb(args, models_pb, im, check_blobs)

    print('Checking models...')
    assert mutils.compare_model(
        _run_cfg_func, _run_pb_func, test_img, check_blobs)


def convert_to_pb(args, net, blobs, part_name='net', input_blobs=[]):
    pb_net = core.Net('')
    pb_net.Proto().op.extend(copy.deepcopy(net.op))

    pb_net.Proto().external_input.extend(
        copy.deepcopy(net.external_input))
    pb_net.Proto().external_output.extend(
        copy.deepcopy(net.external_output))
    pb_net.Proto().type = args.net_execution_type
    pb_net.Proto().num_workers = 1 if args.net_execution_type == 'simple' else 4

    # Reset the device_option, change to unscope name and replace python operators
    convert_net(args, pb_net.Proto(), blobs)

    # add operators for bbox
    add_bbox_ops(args, pb_net, blobs)

    if args.fuse_af:
        print('Fusing affine channel...')
        pb_net, blobs = mutils.fuse_net_affine(pb_net, blobs)

    if args.use_nnpack:
        mutils.update_mobile_engines(pb_net.Proto())

    # generate init net
    pb_init_net = gen_init_net(pb_net, blobs, input_blobs)

    if args.device == 'gpu':
        [pb_net, pb_init_net] = convert_model_gpu(args, pb_net, pb_init_net)

    pb_net.Proto().name = args.net_name + '_' + part_name
    pb_init_net.Proto().name = args.net_name + '_' + part_name + '_init'

    return pb_net, pb_init_net


def main():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    logger.info('Conerting model with config:')
    logger.info(pprint.pformat(cfg))

    models_pb = {}
    # load model from cfg
    model, blobs = load_model(args)

    input_net = ['data', 'im_info']
    models_pb['net'] = convert_to_pb(args, model.net.Proto(), blobs, input_blobs=input_net)

    if cfg.MODEL.MASK_ON:
        models_pb['mask_net'] = convert_to_pb(args, model.mask_net.Proto(), blobs, part_name='mask_net')

    if cfg.MODEL.KEYPOINTS_ON:
        models_pb['keypoint_net'] = convert_to_pb(args, model.keypoint_net.Proto(), blobs, part_name='keypoint_net')

    for (pb_net, pb_init_net) in models_pb.values():
        _save_models(pb_net, pb_init_net, args)

    if args.test_img is not None:
        verify_model(args, models_pb, args.test_img)

if __name__ == '__main__':
    main()
