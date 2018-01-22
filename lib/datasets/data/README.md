# Setting Up Datasets

This directory contains symlinks to data locations.

## Creating Symlinks for COCO

Symlink the COCO dataset:

```
ln -s /path/to/coco $DETECTRON/lib/datasets/data/coco
```

We assume that your local COCO dataset copy at `/path/to/coco` has the following directory structure:

```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

If that is not the case, you may need to do something similar to:

```
mkdir -p $DETECTRON/lib/datasets/data/coco
ln -s /path/to/coco_train2014 $DETECTRON/lib/datasets/data/coco/
ln -s /path/to/coco_val2014 $DETECTRON/lib/datasets/data/coco/
ln -s /path/to/json/annotations $DETECTRON/lib/datasets/data/coco/annotations
```

### COCO Minival Annotations

Our custom `minival` and `valminusminival` annotations are available for download [here](https://s3-us-west-2.amazonaws.com/detectron/coco/coco_annotations_minival.tgz).
Please note that `minival` is exactly equivalent to the recently defined 2017 `val` set.
Similarly, the union of `valminusminival` and the 2014 `train` is exactly equivalent to the 2017 `train` set. To complete installation of the COCO dataset, you will need to copy the `minival` and `valminusminival` json annotation files to the `coco/annotations` directory referenced above.

## Creating Symlinks for PASCAL VOC

Symlink the PASCAL VOC dataset:

```
# VOC 2007
mkdir -p $DETECTRON/lib/datasets/data/VOC2007
ln -s /path/to/VOC2007/JPEG/images $DETECTRON/lib/datasets/data/VOC2007/JPEGImages
ln -s /path/to/VOC2007/json/annotations $DETECTRON/lib/datasets/annotations
ln -s /path/to/VOC2007/devkit $DETECTRON/lib/datasets/VOCdevkit2007

# VOC 2012
mkdir -p $DETECTRON/lib/datasets/data/VOC2012
ln -s /path/to/VOC2012/JPEG/images $DETECTRON/lib/datasets/data/VOC2012/JPEGImages
ln -s /path/to/VOC2012/json/annotations $DETECTRON/lib/datasets/annotations
ln -s /path/to/VOC2012/devkit $DETECTRON/lib/datasets/VOCdevkit2012
```

## Creating Symlinks for Cityscapes:

Symlink the Cityscapes dataset:

```
mkdir -p $DETECTRON/lib/datasets/data/cityscapes
ln -s /path/to/cityscapes/images $DETECTRON/lib/datasets/data/cityscapes/images
ln -s /path/to/cityscapes/json/annotations $DETECTRON/lib/datasets/data/cityscapes/annotations
ln -s /path/to/cityscapes/root/dir $DETECTRON/lib/datasets/data/cityscapes/raw
```
