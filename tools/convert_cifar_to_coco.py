#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @author: Ouail Bendidi
    Created on Wed Jan 31 16:45:36 2018

    Script to convert Cifar-100 dataset into coco format,
    with empty boxes and empty segmentation , each image maps to one
    annotation with it's labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import os
import png
import io
import sys
import cv2
import pprint
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cifar-100 to COCO'
    )
    parser.add_argument(
        '--cifar-path',
        dest='cifar',
        help='path to cifar dataset',
        default=None,
        type=str
    )

    parser.add_argument(
        '--save-path',
        dest='savePath',
        help='path where to save generated coco format dataset',
        default=None,
        type=str
    )
    parser.add_argument(
        '--coarse',
        dest='coarse',
        help='use coarse labels instead of fine labels',
        action='store_true'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def add_annotations(annotID, labelID, imageID, area=0,bbox=[],seg=[]):
    return {
        u"area": area,
        u"id": annotID,
        u"category_id": labelID,
        u"ignore": 0,
        u"segmentation": seg,
        u"image_id": imageID,
        u"bbox":  bbox,
        u"iscrowd": 0
    }


def add_image(imageID, fileName, Width, Height):
    return {
        u"id": imageID,
        u"file_name": fileName,
        u"width": Width,
        u"height": Height
    }

def main(arg):

    save_path_annotation = os.path.join(arg.savePath, "annotations/")
    if not os.path.exists(save_path_annotation):
        os.makedirs(save_path_annotation)

    labels = pickle.load(open(os.path.join(arg.cifar, 'meta'), 'rb'))
    if arg.coarse:
        id_to_label = {name:i+1 for i,name in
                       enumerate(labels["coarse_label_names"])}
    else :
        id_to_label = {name:i+1 for i,name in
                       enumerate(labels["fine_label_names"])}
    pprint.pprint(id_to_label)

    for batch in ('test', 'train'):
        annotID = 0

        fpath = os.path.join(arg.cifar, batch)

        if batch == 'test':batch="val"

        save_path_image = os.path.join(arg.savePath, "coco_{}2014".format(batch))
        anot_file_name = os.path.join(
            save_path_annotation, "instances_{}2014.json".format(batch))
        if not os.path.exists(save_path_image):
            os.makedirs(save_path_image)

        json_data = {
            u"categories": [],
            u"annotations": [],
            u"images": [],
            u"type": "classification"}

        f = open(fpath, 'rb')

        d = pickle.load(f)
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v

        d = d_decoded
        f.close()

        for i, filename in enumerate(d['filenames']):

            output = "{} : {}".format(batch, filename)
            sys.stdout.write("\r\x1b[K" + output)
            sys.stdout.flush()

            if arg.coarse:
                labelID = id_to_label[labels['coarse_label_names']
                                      [d['coarse_labels'][i]]]
            else :
                labelID = id_to_label[labels['fine_label_names']
                                      [d['fine_labels'][i]]]

            json_data["annotations"].append(
                        add_annotations(annotID,
                                        labelID,
                                        annotID,
                                        area=1)
                        )


            json_data["images"].append(
                add_image(annotID, filename, 32, 32)
                                       )
            img = d['data'][i]
            out_name = os.path.join(save_path_image,filename)
            img = img.reshape((32, 32, 3), order='F').swapaxes(0,1)[:, :, (2, 1, 0)]
            cv2.imwrite(out_name,img)

            annotID += 1
        for key in id_to_label:
            json_data["categories"].append(
                {u"id": id_to_label[key],
                 u"name": key,
                 u"supercategory": None})
        with io.open(anot_file_name, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(json_data, ensure_ascii=False))
        print("\n")
    print("num of labels is ",len(id_to_label))

if __name__ == '__main__':
    args = parse_args()
    main(args)
