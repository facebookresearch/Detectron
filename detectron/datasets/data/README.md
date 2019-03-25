# Setting Up Datasets

This directory contains symlinks to data locations.

## Creating Symlinks for COCO

Symlink the COCO dataset:

```
ln -s /path/to/coco $DETECTRON/detectron/datasets/data/coco
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
mkdir -p $DETECTRON/detectron/datasets/data/coco
ln -s /path/to/coco_train2014 $DETECTRON/detectron/datasets/data/coco/coco_train2014
ln -s /path/to/coco_val2014 $DETECTRON/detectron/datasets/data/coco/coco_val2014
ln -s /path/to/json/annotations $DETECTRON/detectron/datasets/data/coco/annotations
```

### COCO Minival Annotations

Our custom `minival` and `valminusminival` annotations are available for download [here](https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz).
Please note that `minival` is exactly equivalent to the recently defined 2017 `val` set.
Similarly, the union of `valminusminival` and the 2014 `train` is exactly equivalent to the 2017 `train` set. To complete installation of the COCO dataset, you will need to copy the `minival` and `valminusminival` json annotation files to the `coco/annotations` directory referenced above.

## Creating Symlinks for PASCAL VOC

We assume that your symlinked `detectron/datasets/data/VOC<year>` directory has the following structure:

```
VOC<year>
|_ JPEGImages
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
|  |_ voc_<year>_train.json
|  |_ voc_<year>_val.json
|  |_ ...
|_ VOCdevkit<year>
```

Create symlinks for `VOC<year>`:

```
mkdir -p $DETECTRON/detectron/datasets/data/VOC<year>
ln -s /path/to/VOC<year>/JPEGImages $DETECTRON/detectron/datasets/data/VOC<year>/JPEGImages
ln -s /path/to/VOC<year>/json/annotations $DETECTRON/detectron/datasets/data/VOC<year>/annotations
ln -s /path/to/VOC<year>/devkit $DETECTRON/detectron/datasets/data/VOC<year>/VOCdevkit<year>
```

### PASCAL VOC Annotations in COCO Format

We expect PASCAL VOC annotations converted to COCO json format, which are available for download [here](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip ).

## Creating Symlinks for Cityscapes:

We assume that your symlinked `detectron/datasets/data/cityscapes` directory has the following structure:

```
cityscapes
|_ images
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
|  |_ instanceonly_gtFile_train.json
|  |_ ...
|_ raw
   |_ gtFine
   |_ ...
   |_ README.md
```

Create symlinks for `cityscapes`:

```
mkdir -p $DETECTRON/detectron/datasets/data/cityscapes
ln -s /path/to/cityscapes/images $DETECTRON/detectron/datasets/data/cityscapes/images
ln -s /path/to/cityscapes/json/annotations $DETECTRON/detectron/datasets/data/cityscapes/annotations
ln -s /path/to/cityscapes/root $DETECTRON/detectron/datasets/data/cityscapes/raw
```

### Cityscapes Annotations in COCO Format

We expect Cityscapes annotations converted to COCO json format, which we will make available for download soon.
