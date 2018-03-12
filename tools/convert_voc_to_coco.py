#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    @author: Ouail Bendidi

    Script to Convert Pascal Voc format dataset to coco format

"""


import random
import math
import io
import json
import os
import sys
import numpy as np
import pprint
import cv2
import os
import xmltodict
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert voc dataset to coco format'
    )
    parser.add_argument(
        '--voc',
        dest='voc',
        help='path to voc dataset',
        default=None,
        type=str
    )

    parser.add_argument(
        '--coco',
        dest='coco',
        help='path where to save generated coco format dataset',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def convert(xml_file, xml_attribs=True):
    with open(xml_file, "rb") as f:
        d = xmltodict.parse(f, xml_attribs=xml_attribs)
        return d


def add_annotations(annotID, labelID, imageID, area=0, bbox=[], seg=[]):
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


def split_train_test(allFiles, ratio=0.8):
    n = len(allFiles)
    random.shuffle(allFiles)

    return {"train": allFiles[:int(n * ratio)], "val": allFiles[int(n * ratio):]}


def main(arg):

    xml_path = os.path.join(arg.voc, "Annotations")
    img_path = os.path.join(arg.voc, "JPEGImages")
    save_path_annotation = os.path.join(arg.coco, "annotations/")

    if not os.path.exists(save_path_annotation):
        os.makedirs(save_path_annotation)

    allXmlFiles = sorted(os.listdir(xml_path))
    trainValXmlFiles = split_train_test(allXmlFiles, ratio=0.8)
    id_to_label = {}

    for Set in ["train", "val"]:

        imageID = 0
        annotID = 0

        save_path_image = os.path.join(arg.coco, "coco_{}2014".format(Set))
        anot_file_name = os.path.join(
            save_path_annotation, "instances_{}2014.json".format(Set))

        if not os.path.exists(save_path_image):
            os.makedirs(save_path_image)

        json_data = {
            u"categories": [],
            u"annotations": [],
            u"images": [],
            u"type": "instance"}

        for fc in trainValXmlFiles[Set]:

            output = "{} : {}".format(Set, fc)
            sys.stdout.write("\r\x1b[K" + output)
            sys.stdout.flush()

            dic = convert(os.path.join(xml_path, fc))

            im_name = dic['annotation']['path'].split("/")[-1]

            if "object" in dic["annotation"].keys():
                if not isinstance(dic['annotation']["object"], list):
                    dic['annotation']["object"] = [dic['annotation']["object"]]
                for x in dic['annotation']["object"]:

                    if len(id_to_label.keys()) == 0 or x["name"] not in id_to_label.keys():
                        if len(id_to_label.values()) == 0:
                            id_to_label[str(x["name"])] = 1
                        else:
                            id_to_label[x["name"]] = max(
                                id_to_label.values()) + 1

                    json_data["annotations"].append(
                        add_annotations(annotID,
                                        id_to_label[x["name"]],
                                        imageID,
                                        area=(int(float(x['bndbox']['xmax'])) -
                                              int(float(x['bndbox']['xmin']))) *
                                        (int(float(x['bndbox']['ymax'])) -
                                         int(float(x['bndbox']['ymin']))),
                                        bbox=[int(float(x['bndbox']['xmin'])),
                                              int(float(x['bndbox']['ymin'])),
                                              int(float(float(x['bndbox']['xmax']))) -
                                              int(float(x['bndbox']['xmin'])),
                                              int(float(x['bndbox']['ymax'])) -
                                              int(float(x['bndbox']['ymin']))]
                                        ))
                    annotID += 1

            json_data["images"].append(
                add_image(imageID,
                          dic['annotation']['path'].split("/")[-1],
                          int(dic['annotation']['size']["width"]),
                          int(dic['annotation']['size']["height"])))

            shutil.copy(os.path.join(img_path, im_name),
                        os.path.join(save_path_image,
                                     im_name))
            imageID += 1
        for key in id_to_label:
            json_data["categories"].append(
                {u"id": id_to_label[key],
                 u"name": key,
                 u"supercategory": None})

        with io.open(anot_file_name, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(json_data, ensure_ascii=False))
        print("\n")
    pprint.pprint(id_to_label)
    print("num of labels is ", len(id_to_label))


if __name__ == '__main__':
    args = parse_args()
    main(args)
