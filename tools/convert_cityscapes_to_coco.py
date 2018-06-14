from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys

import cityscapesscripts.evaluation.instances2dict as cs

import detectron.utils.segms as segms_util
import detectron.utils.boxes as bboxs_util


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def convert_coco_stuff_mat(data_dir, out_dir):
    """Convert to png and save json with path. This currently only contains
    the segmentation labels for objects+stuff in cocostuff - if we need to
    combine with other labels from original COCO that will be a TODO."""
    sets = ['train', 'val']
    categories = []
    json_name = 'coco_stuff_%s.json'
    ann_dict = {}
    for data_set in sets:
        file_list = os.path.join(data_dir, '%s.txt')
        images = []
        with open(file_list % data_set) as f:
            for img_id, img_name in enumerate(f):
                img_name = img_name.replace('coco', 'COCO').strip('\n')
                image = {}
                mat_file = os.path.join(
                    data_dir, 'annotations/%s.mat' % img_name)
                data = h5py.File(mat_file, 'r')
                labelMap = data.get('S')
                if len(categories) == 0:
                    labelNames = data.get('names')
                    for idx, n in enumerate(labelNames):
                        categories.append(
                            {"id": idx, "name": ''.join(chr(i) for i in data[
                                n[0]])})
                    ann_dict['categories'] = categories
                scipy.misc.imsave(
                    os.path.join(data_dir, img_name + '.png'), labelMap)
                image['width'] = labelMap.shape[0]
                image['height'] = labelMap.shape[1]
                image['file_name'] = img_name
                image['seg_file_name'] = img_name
                image['id'] = img_id
                images.append(image)
        ann_dict['images'] = images
        print("Num images: %s" % len(images))
        with open(os.path.join(out_dir, json_name % data_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'image_train',
        'image_val',

        # 'gtFine_test',
        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'gtFine/train',
        'gtFine/val',
        # 'gtFine_trainvaltest/gtFine/test',

        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    add_instancesonly = ['__background__',
                              'person',
                              'car'
                              ]
    category_instancesonly = ['__background__',
        'guard rail',
        'car',
        'dashed',
        'solid',
        'solid solid',
        'dashed dashed',
        'dashed-solid',
        'solid-dashed',
        'yellow dashed',
        'yellow solid',
        'yellow solid solid',
        'yellow dashed dashed',
        'yellow dashed-solid',
        'yellow solid-dashed',
        'boundary'
    ]
    # category_instancesonly = ['__background__',
    #                           'ego vehicle',
    #                           'rectification border',
    #                           'out of roi',
    #                           'static',
    #                           'dynamic',
    #                           'ground',
    #                           'road',
    #                           'sidewalk',
    #                           'parking',
    #                           'rail track',
    #                           'building',
    #                           'wall',
    #                           'fence',
    #                           'guard rail',
    #                           'bridge',
    #                           'tunnel',
    #                           'pole',
    #                           'polegroup',
    #                           'traffic light',
    #                           'traffic sign',
    #                           'vegetation',
    #                           'terrain',
    #                           'sky',
    #                           'person',
    #                           'rider',
    #                           'car',
    #                           'truck',
    #                           'bus',
    #                           'caravan',
    #                           'trailer',
    #                           'train',
    #                           'motorcycle',
    #                           'bicycle',
    # ]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, sub_dirs, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    sub_file_name = filename.split('_')
                    image['file_name'] = os.path.join(sub_file_name[0], '_'.join(sub_file_name[:-2]) + '_leftImg8bit.png')
                    image['seg_file_name'] = '_'.join(filename.split('_')[:-1]) + '_instanceIds.png'
                    images.append(image)

                    fullname = os.path.join(root, image['seg_file_name'])
                    print ("fullname:" + fullname)
                    objects = cs.instances2dict_with_polygons(
                        [fullname], verbose=False)[fullname]

                    for object_cls in objects:
                        # if object_cls not in add_instancesonly:
                        #     continue

                        if object_cls not in category_instancesonly:
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:
                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            index = category_instancesonly.index(object_cls)  # + 184
                            good_area = [p for p in obj['contours'] if len(p) > 4]

                            if len(good_area) == 0:
                                print('Warning: invalid contours.')
                                continue  # skip non-instance categories

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = good_area

                            ann['category_id'] = index
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount']
                            ann['bbox'] = bboxs_util.xyxy_to_xywh(
                                segms_util.polys_to_boxes(
                                    [ann['segmentation']])).tolist()[0]

                            annotations.append(ann)

        ann_dict['images'] = images

        categories = []
        for index, value in enumerate(category_instancesonly):
            categories.append({"id": index, "name": value})
        categories = categories[1:]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    # args.datadir = "/media/administrator/deeplearning/dataset/cityscape"
    # args.outdir = "/media/administrator/deeplearning/dataset/cityscape/output"
    args.datadir = "/media/administrator/deeplearning/self-labels"
    args.outdir = "/media/administrator/deeplearning/self-labels/output"
    # args.datadir = "/media/administrator/deeplearning/dataset/test_cityscape"
    # args.outdir = "/media/administrator/deeplearning/dataset/test_cityscape/output"
    convert_cityscapes_instance_only(args.datadir, args.outdir)
