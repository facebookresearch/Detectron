# Detectron Transfer Learning with PASCAL VOC 2007 dataset

** Detectron implemented  several object detecton algorithms. All the algorithms are trained on coco 2014 data set which has 80 categories. I want to fine tune the faster-rcnn with FPN on pascal voc 2007 dataset which has only 20 categories. The same way can be used to fine tune your own model on a new dataset **

## 1. Setup caffe2 and Detectron and run the Detectron demo successfully.
I will refer the Detectron directory as $DETECTRON

## 2. Download the pre-trained model
The code will download the models automatically. But my internet is slow and I'd like to download them before I run the code.
Becase I'm gonna using the ResNet-50 as the backbone, so I need to download the ResNet and faster_rcnn_R-50-FPN model.
```
wget https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl /tmp/detectron/detectron-download-cache/ImageNetPretrained/MSRA/R-50.pkl
wget https://s3-us-west-2.amazonaws.com/detectron/36225732/12_2017_baselines/mask_rcnn_R-50-FPN_2x.yaml.08_43_08.gDqBz9zS/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
```

## 3. Prepare configuration file.
### a. Copy the sample configure file from $DETECTRON/configs/getting_started
```
cd $DETECTORN
mkdir experiments && cd experiments
cp ../configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml e2e_faster_rcnn_resnet-50-FPN_pascal2007.yaml
```
### b. Change the configuration file
```
MODEL:
  TYPE: generalized_rcnn
  CONV_BODY: FPN.add_fpn_ResNet50_conv5_body
  NUM_CLASSES: 21
  FASTER_RCNN: True
```
The pascal voc 2007 has only 20 classes plus one background class. So the NUM_CLASSES is set to 21.

```
TRAIN:
  SNAPSHOT_ITERS: 5000
  WEIGHTS: /tmp/detectron/detectron-download-cache/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
  DATASETS: ('voc_2007_train',)
```
Change the WEIGHTS value to where you just place in step 2.

## 4. Download pascal voc2007 and coco format annotations.
Refer [data readme file](https://github.com/facebookresearch/Detectron/blob/master/lib/datasets/data/README.md) to prepare the pascal data set.

The code support pascal data set has bug, the code in $DETECTRON/lib/datasets/dataset_catalog.py should be changed as following:
```
    'voc_2007_train': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/pascal_train2007.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/pascal_test2007.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },

```

## 5. Change the cls_score and bbox_pred name to prevent error when train_net.py load weights
In lib/modeling/fast_rcnn_heads.py, change all cls_score to cls_score_voc, bbox_pred to bbox_pred_voc.

## 6. Run command to begin training.
```
python2 tools/train_net.py --cfg experiments/e2e_faster_rcnn_resnet-50-FPN_pascal2007.yaml  OUTPUT_DIR experiments/output
```

## 7. Copy the final model just trained.
```
mkdir -p /tmp/detectron-download-cache/voc2007/
cp experiments/output/train/voc_2007_train/generalized_rcnn/model_iter49999.pkl /tmp/detectron-download-cache/voc2007/model_final.pkl

```
## 8. Infer some images psacal 2007 from test dataset
```
python2 tools/infer_simple.py --cfg experiments/e2e_faster_rcnn_resnet-50-FPN_pascal2007.yaml \
    --output-dir /tmp/detectron-visualizations --wts /tmp/detectron-download-cache/voc2007/model_final.pkl \
    demo2
```
Unfortunately, I found all person is labeld as bird. This maybe caused by the json dataset is not corrrectly converted.

## 9. Run test_net.py on pascal 2007 test dataset.
```
python2 tools/test_net.py \
    --cfg experiments/e2e_faster_rcnn_resnet-50-FPN_pascal2007.yaml \
    TEST.WEIGHTS /tmp/detectron-download-cache/voc2007/model_final.pkl \
    NUM_GPUS 1
```
The test report the AP and mAP:
```
INFO voc_dataset_evaluator.py: 127: AP for aeroplane = 0.8095
INFO voc_dataset_evaluator.py: 127: AP for bicycle = 0.8042
INFO voc_dataset_evaluator.py: 127: AP for bird = 0.7086
INFO voc_dataset_evaluator.py: 127: AP for boat = 0.6418
INFO voc_dataset_evaluator.py: 127: AP for bottle = 0.6861
INFO voc_dataset_evaluator.py: 127: AP for bus = 0.8822
INFO voc_dataset_evaluator.py: 127: AP for car = 0.8794
INFO voc_dataset_evaluator.py: 127: AP for cat = 0.8621
INFO voc_dataset_evaluator.py: 127: AP for chair = 0.5876
INFO voc_dataset_evaluator.py: 127: AP for cow = 0.7799
INFO voc_dataset_evaluator.py: 127: AP for diningtable = 0.7404
INFO voc_dataset_evaluator.py: 127: AP for dog = 0.8497
INFO voc_dataset_evaluator.py: 127: AP for horse = 0.8855
INFO voc_dataset_evaluator.py: 127: AP for motorbike = 0.7912
INFO voc_dataset_evaluator.py: 127: AP for person = 0.7931
INFO voc_dataset_evaluator.py: 127: AP for pottedplant = 0.5142
INFO voc_dataset_evaluator.py: 127: AP for sheep = 0.7950
INFO voc_dataset_evaluator.py: 127: AP for sofa = 0.7457
INFO voc_dataset_evaluator.py: 127: AP for train = 0.7956
INFO voc_dataset_evaluator.py: 127: AP for tvmonitor = 0.6960
INFO voc_dataset_evaluator.py: 130: Mean AP = 0.7624

```
 