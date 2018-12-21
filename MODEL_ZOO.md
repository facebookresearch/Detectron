# Detectron Model Zoo and Baselines

## Introduction

This file documents a large collection of baselines trained with Detectron, primarily in late December 2017. We refer to these results as the *12_2017_baselines*. All configurations for these baselines are located in the `configs/12_2017_baselines` directory. The tables below provide results and useful statistics about training and inference. Links to the trained models as well as their output are provided. Unless noted differently below (see "Notes" under each table), the following common settings are used for all training and inference runs.

#### Common Settings and Notes

- All baselines were run on [Big Basin](https://code.facebook.com/posts/1835166200089399/introducing-big-basin) servers with 8 NVIDIA Tesla P100 GPU accelerators (with 16GB GPU memory, CUDA 8.0, and cuDNN 6.0.21).
- All baselines were trained using 8 GPU data parallel sync SGD with a minibatch size of either 8 or 16 images (see the *im/gpu* column).
- For training, only horizontal flipping data augmentation was used.
- For inference, no test-time augmentations (e.g., multiple scales, flipping) were used.
- All models were trained on the union of `coco_2014_train` and `coco_2014_valminusminival`, which is exactly equivalent to the recently defined `coco_2017_train` dataset.
- All models were tested on the `coco_2014_minival` dataset, which is exactly equivalent to the recently defined `coco_2017_val` dataset.
- Inference times are often expressed as "*X* + *Y*", in which *X* is time taken in reasonably well-optimized GPU code and *Y* is time taken in unoptimized CPU code. (The CPU code time could be reduced substantially with additional engineering.)
- Inference results for boxes, masks, and keypoints ("kps") are provided in the [COCO json format](http://cocodataset.org/#format-data).
- The *model id* column is provided for ease of reference.
- To check downloaded file integrity: for any download URL on this page, simply append `.md5sum` to the URL to download the file's md5 hash.
- All models and results below are on the [COCO dataset](http://cocodataset.org).
- Baseline models and results for the [Cityscapes dataset](https://www.cityscapes-dataset.com/) are coming soon!

#### Training Schedules

We use three training schedules, indicated by the *lr schd* column in the tables below.

- **1x**: For minibatch size 16, this schedule starts at a LR of 0.02 and is decreased by a factor of * 0.1 after 60k and 80k iterations and finally terminates at 90k iterations. This schedules results in 12.17 epochs over the 118,287 images in `coco_2014_train` union `coco_2014_valminusminival` (or equivalently, `coco_2017_train`).
- **2x**: Twice as long as the 1x schedule with the LR change points scaled proportionally.
- **s1x** ("stretched 1x"): This schedule scales the 1x schedule by roughly 1.44x, but also extends the duration of the first learning rate. With a minibatch size of 16, it reduces the LR by * 0.1 at 100k and 120k iterations, finally ending after 130k iterations.

All training schedules also use a 500 iteration linear learning rate warm up. When changing the minibatch size between 8 and 16 images, we adjust the number of SGD iterations and the base learning rate according to the principles outlined in our paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).

#### License

All models available for download through this document are licensed under the [Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

#### ImageNet Pretrained Models

The backbone models pretrained on ImageNet are available in the format used by Detectron. Unless otherwise noted, these models are trained on the standard ImageNet-1k dataset.

- [R-50.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl): converted copy of MSRA's original ResNet-50 model
- [R-101.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl): converted copy of MSRA's original ResNet-101 model
- [X-101-64x4d.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl): converted copy of FB's original ResNeXt-101-64x4d model trained with Torch7
- [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB
- [X-152-32x8d-IN5k.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl): ResNeXt-152-32x8d model **trained on ImageNet-5k** with Caffe2 at FB (see our [ResNeXt paper](https://arxiv.org/abs/1611.05431) for details on ImageNet-5k)

#### Log Files

[Training and inference logs](https://dl.fbaipublicfiles.com/detectron/logs/model_zoo_12_2017_baseline_logs.tgz) are available for most models in the model zoo.

## Proposal, Box, and Mask Detection Baselines

### RPN Proposal Baselines

<table><tbody>
<!-- START RPN TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop.<br/>AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>4.3</sub></sup></td>
<td align="right"><sup><sub>0.187</sub></sup></td>
<td align="right"><sup><sub>4.7</sub></sup></td>
<td align="right"><sup><sub>0.113</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>51.6</sub></sup></td>
<td align="right"><sup><sub>35998355</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L/output/train/coco_2014_train%3Acoco_2014_valminusminival/rpn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L/output/test/coco_2014_train/rpn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L/output/test/coco_2014_valminusminival/rpn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L/output/test/coco_2014_minival/rpn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.416</sub></sup></td>
<td align="right"><sup><sub>10.4</sub></sup></td>
<td align="right"><sup><sub>0.080</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>57.2</sub></sup></td>
<td align="right"><sup><sub>35998814</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179/output/test/coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179/output/test/coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179/output/test/coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.1</sub></sup></td>
<td align="right"><sup><sub>0.503</sub></sup></td>
<td align="right"><sup><sub>12.6</sub></sup></td>
<td align="right"><sup><sub>0.108</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>58.2</sub></sup></td>
<td align="right"><sup><sub>35998887</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35998887/12_2017_baselines/rpn_R-101-FPN_1x.yaml.08_07_07.vzhHEs0V/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998887/12_2017_baselines/rpn_R-101-FPN_1x.yaml.08_07_07.vzhHEs0V/output/test/coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998887/12_2017_baselines/rpn_R-101-FPN_1x.yaml.08_07_07.vzhHEs0V/output/test/coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998887/12_2017_baselines/rpn_R-101-FPN_1x.yaml.08_07_07.vzhHEs0V/output/test/coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>11.5</sub></sup></td>
<td align="right"><sup><sub>1.395</sub></sup></td>
<td align="right"><sup><sub>34.9</sub></sup></td>
<td align="right"><sup><sub>0.292</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>59.4</sub></sup></td>
<td align="right"><sup><sub>35998956</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35998956/12_2017_baselines/rpn_X-101-64x4d-FPN_1x.yaml.08_08_41.Seh0psKz/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998956/12_2017_baselines/rpn_X-101-64x4d-FPN_1x.yaml.08_08_41.Seh0psKz/output/test/coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998956/12_2017_baselines/rpn_X-101-64x4d-FPN_1x.yaml.08_08_41.Seh0psKz/output/test/coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998956/12_2017_baselines/rpn_X-101-64x4d-FPN_1x.yaml.08_08_41.Seh0psKz/output/test/coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>11.6</sub></sup></td>
<td align="right"><sup><sub>1.102</sub></sup></td>
<td align="right"><sup><sub>27.6</sub></sup></td>
<td align="right"><sup><sub>0.222</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>59.5</sub></sup></td>
<td align="right"><sup><sub>36760102</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36760102/12_2017_baselines/rpn_X-101-32x8d-FPN_1x.yaml.06_00_16.RWeBAniO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760102/12_2017_baselines/rpn_X-101-32x8d-FPN_1x.yaml.06_00_16.RWeBAniO/output/test/coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760102/12_2017_baselines/rpn_X-101-32x8d-FPN_1x.yaml.06_00_16.RWeBAniO/output/test/coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760102/12_2017_baselines/rpn_X-101-32x8d-FPN_1x.yaml.06_00_16.RWeBAniO/output/test/coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
</tr>
<!-- END RPN TABLE -->
</tbody></table>

**Notes:**

- Inference time only includes RPN proposal generation.
- "prop. AR" is proposal average recall at 1000 proposals per image.
- Proposal download links ("props"): "1" is `coco_2014_train`; "2" is `coco_2014_valminusminival`; and "3" is `coco_2014_minival`.

### Fast & Mask R-CNN Baselines Using Precomputed RPN Proposals

<table><tbody>
<!-- START 2-STAGE TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop.<br/>AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.0</sub></sup></td>
<td align="right"><sup><sub>0.456</sub></sup></td>
<td align="right"><sup><sub>22.8</sub></sup></td>
<td align="right"><sup><sub>0.241&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>34.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36224013</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36224013/12_2017_baselines/fast_rcnn_R-50-C4_1x.yaml.08_22_00.vHd5BeBP/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224013/12_2017_baselines/fast_rcnn_R-50-C4_1x.yaml.08_22_00.vHd5BeBP/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.0</sub></sup></td>
<td align="right"><sup><sub>0.453</sub></sup></td>
<td align="right"><sup><sub>45.3</sub></sup></td>
<td align="right"><sup><sub>0.241&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>35.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36224046</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36224046/12_2017_baselines/fast_rcnn_R-50-C4_2x.yaml.08_22_57.XFxNqEnL/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224046/12_2017_baselines/fast_rcnn_R-50-C4_2x.yaml.08_22_57.XFxNqEnL/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.0</sub></sup></td>
<td align="right"><sup><sub>0.285</sub></sup></td>
<td align="right"><sup><sub>7.1</sub></sup></td>
<td align="right"><sup><sub>0.076&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>36.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36225147</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36225147/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml.08_39_09.L3obSdQ2/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225147/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml.08_39_09.L3obSdQ2/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.0</sub></sup></td>
<td align="right"><sup><sub>0.287</sub></sup></td>
<td align="right"><sup><sub>14.4</sub></sup></td>
<td align="right"><sup><sub>0.077&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>36.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36225249</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36225249/12_2017_baselines/fast_rcnn_R-50-FPN_2x.yaml.08_40_18.zoChak1f/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225249/12_2017_baselines/fast_rcnn_R-50-FPN_2x.yaml.08_40_18.zoChak1f/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.448</sub></sup></td>
<td align="right"><sup><sub>11.2</sub></sup></td>
<td align="right"><sup><sub>0.102&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>38.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36228880</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36228880/12_2017_baselines/fast_rcnn_R-101-FPN_1x.yaml.09_25_03.tZuHkSpl/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36228880/12_2017_baselines/fast_rcnn_R-101-FPN_1x.yaml.09_25_03.tZuHkSpl/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.449</sub></sup></td>
<td align="right"><sup><sub>22.5</sub></sup></td>
<td align="right"><sup><sub>0.103&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>39.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36228933</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36228933/12_2017_baselines/fast_rcnn_R-101-FPN_2x.yaml.09_26_27.jkOUTrrk/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36228933/12_2017_baselines/fast_rcnn_R-101-FPN_2x.yaml.09_26_27.jkOUTrrk/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.3</sub></sup></td>
<td align="right"><sup><sub>0.994</sub></sup></td>
<td align="right"><sup><sub>49.7</sub></sup></td>
<td align="right"><sup><sub>0.292&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>40.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36226250</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36226250/12_2017_baselines/fast_rcnn_X-101-64x4d-FPN_1x.yaml.08_54_22.u0LaxQsC/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36226250/12_2017_baselines/fast_rcnn_X-101-64x4d-FPN_1x.yaml.08_54_22.u0LaxQsC/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.3</sub></sup></td>
<td align="right"><sup><sub>0.980</sub></sup></td>
<td align="right"><sup><sub>98.0</sub></sup></td>
<td align="right"><sup><sub>0.291&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>39.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36226326</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36226326/12_2017_baselines/fast_rcnn_X-101-64x4d-FPN_2x.yaml.08_55_54.2F7MP1CD/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36226326/12_2017_baselines/fast_rcnn_X-101-64x4d-FPN_2x.yaml.08_55_54.2F7MP1CD/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.721</sub></sup></td>
<td align="right"><sup><sub>36.1</sub></sup></td>
<td align="right"><sup><sub>0.217&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>40.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37119777</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37119777/12_2017_baselines/fast_rcnn_X-101-32x8d-FPN_1x.yaml.06_38_03.d5N36egm/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37119777/12_2017_baselines/fast_rcnn_X-101-32x8d-FPN_1x.yaml.06_38_03.d5N36egm/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Fast</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.720</sub></sup></td>
<td align="right"><sup><sub>72.0</sub></sup></td>
<td align="right"><sup><sub>0.217&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>39.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37121469</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37121469/12_2017_baselines/fast_rcnn_X-101-32x8d-FPN_2x.yaml.07_03_53.EPrHk63L/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37121469/12_2017_baselines/fast_rcnn_X-101-32x8d-FPN_2x.yaml.07_03_53.EPrHk63L/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.466</sub></sup></td>
<td align="right"><sup><sub>23.3</sub></sup></td>
<td align="right"><sup><sub>0.252&nbsp;+&nbsp;0.020</sub></sup></td>
<td align="right"><sup><sub>35.5</sub></sup></td>
<td align="right"><sup><sub>31.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36224121</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36224121/12_2017_baselines/mask_rcnn_R-50-C4_1x.yaml.08_24_37.wdU8r5Jo/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224121/12_2017_baselines/mask_rcnn_R-50-C4_1x.yaml.08_24_37.wdU8r5Jo/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224121/12_2017_baselines/mask_rcnn_R-50-C4_1x.yaml.08_24_37.wdU8r5Jo/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.464</sub></sup></td>
<td align="right"><sup><sub>46.4</sub></sup></td>
<td align="right"><sup><sub>0.253&nbsp;+&nbsp;0.019</sub></sup></td>
<td align="right"><sup><sub>36.9</sub></sup></td>
<td align="right"><sup><sub>32.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36224151</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36224151/12_2017_baselines/mask_rcnn_R-50-C4_2x.yaml.08_25_34.RSN5CVSH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224151/12_2017_baselines/mask_rcnn_R-50-C4_2x.yaml.08_25_34.RSN5CVSH/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36224151/12_2017_baselines/mask_rcnn_R-50-C4_2x.yaml.08_25_34.RSN5CVSH/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.9</sub></sup></td>
<td align="right"><sup><sub>0.377</sub></sup></td>
<td align="right"><sup><sub>9.4</sub></sup></td>
<td align="right"><sup><sub>0.082&nbsp;+&nbsp;0.019</sub></sup></td>
<td align="right"><sup><sub>37.3</sub></sup></td>
<td align="right"><sup><sub>33.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36225401</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36225401/12_2017_baselines/mask_rcnn_R-50-FPN_1x.yaml.08_42_04.MocEgrRW/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225401/12_2017_baselines/mask_rcnn_R-50-FPN_1x.yaml.08_42_04.MocEgrRW/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225401/12_2017_baselines/mask_rcnn_R-50-FPN_1x.yaml.08_42_04.MocEgrRW/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.9</sub></sup></td>
<td align="right"><sup><sub>0.377</sub></sup></td>
<td align="right"><sup><sub>18.9</sub></sup></td>
<td align="right"><sup><sub>0.083&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>37.7</sub></sup></td>
<td align="right"><sup><sub>34.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36225732</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36225732/12_2017_baselines/mask_rcnn_R-50-FPN_2x.yaml.08_43_08.gDqBz9zS/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225732/12_2017_baselines/mask_rcnn_R-50-FPN_2x.yaml.08_43_08.gDqBz9zS/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36225732/12_2017_baselines/mask_rcnn_R-50-FPN_2x.yaml.08_43_08.gDqBz9zS/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.6</sub></sup></td>
<td align="right"><sup><sub>0.539</sub></sup></td>
<td align="right"><sup><sub>13.5</sub></sup></td>
<td align="right"><sup><sub>0.111&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>39.4</sub></sup></td>
<td align="right"><sup><sub>35.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36229407</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36229407/12_2017_baselines/mask_rcnn_R-101-FPN_1x.yaml.09_38_04.zbVPo8ZE/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36229407/12_2017_baselines/mask_rcnn_R-101-FPN_1x.yaml.09_38_04.zbVPo8ZE/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36229407/12_2017_baselines/mask_rcnn_R-101-FPN_1x.yaml.09_38_04.zbVPo8ZE/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.6</sub></sup></td>
<td align="right"><sup><sub>0.537</sub></sup></td>
<td align="right"><sup><sub>26.9</sub></sup></td>
<td align="right"><sup><sub>0.109&nbsp;+&nbsp;0.016</sub></sup></td>
<td align="right"><sup><sub>40.0</sub></sup></td>
<td align="right"><sup><sub>35.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36229740</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36229740/12_2017_baselines/mask_rcnn_R-101-FPN_2x.yaml.09_39_00.Z7O7zOEC/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36229740/12_2017_baselines/mask_rcnn_R-101-FPN_2x.yaml.09_39_00.Z7O7zOEC/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36229740/12_2017_baselines/mask_rcnn_R-101-FPN_2x.yaml.09_39_00.Z7O7zOEC/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.3</sub></sup></td>
<td align="right"><sup><sub>1.036</sub></sup></td>
<td align="right"><sup><sub>51.8</sub></sup></td>
<td align="right"><sup><sub>0.292&nbsp;+&nbsp;0.016</sub></sup></td>
<td align="right"><sup><sub>41.3</sub></sup></td>
<td align="right"><sup><sub>37.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36226382</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36226382/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_1x.yaml.08_56_59.rUCejrBN/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36226382/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_1x.yaml.08_56_59.rUCejrBN/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36226382/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_1x.yaml.08_56_59.rUCejrBN/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.3</sub></sup></td>
<td align="right"><sup><sub>1.035</sub></sup></td>
<td align="right"><sup><sub>103.5</sub></sup></td>
<td align="right"><sup><sub>0.292&nbsp;+&nbsp;0.014</sub></sup></td>
<td align="right"><sup><sub>41.1</sub></sup></td>
<td align="right"><sup><sub>36.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36672114</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36672114/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_2x.yaml.08_58_13.aNWCi3U7/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36672114/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_2x.yaml.08_58_13.aNWCi3U7/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36672114/12_2017_baselines/mask_rcnn_X-101-64x4d-FPN_2x.yaml.08_58_13.aNWCi3U7/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.4</sub></sup></td>
<td align="right"><sup><sub>0.766</sub></sup></td>
<td align="right"><sup><sub>38.3</sub></sup></td>
<td align="right"><sup><sub>0.223&nbsp;+&nbsp;0.017</sub></sup></td>
<td align="right"><sup><sub>41.3</sub></sup></td>
<td align="right"><sup><sub>37.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37121516</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37121516/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_1x.yaml.07_04_58.CbM22DZg/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37121516/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_1x.yaml.07_04_58.CbM22DZg/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37121516/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_1x.yaml.07_04_58.CbM22DZg/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.4</sub></sup></td>
<td align="right"><sup><sub>0.765</sub></sup></td>
<td align="right"><sup><sub>76.5</sub></sup></td>
<td align="right"><sup><sub>0.222&nbsp;+&nbsp;0.014</sub></sup></td>
<td align="right"><sup><sub>40.7</sub></sup></td>
<td align="right"><sup><sub>36.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37121596</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37121596/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_2x.yaml.07_05_48.TL22uFaK/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37121596/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_2x.yaml.07_05_48.TL22uFaK/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37121596/12_2017_baselines/mask_rcnn_X-101-32x8d-FPN_2x.yaml.07_05_48.TL22uFaK/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<!-- END 2-STAGE TABLE -->
</tbody></table>

**Notes:**

- Each row uses precomputed RPN proposals from the corresponding table row above that uses the same backbone.
- Inference time *excludes* proposal generation.

### End-to-End Faster & Mask R-CNN Baselines

<table><tbody>
<!-- START E2E FASTER AND MASK TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop.<br/>AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.3</sub></sup></td>
<td align="right"><sup><sub>0.566</sub></sup></td>
<td align="right"><sup><sub>28.3</sub></sup></td>
<td align="right"><sup><sub>0.167&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>34.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857197</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.3</sub></sup></td>
<td align="right"><sup><sub>0.569</sub></sup></td>
<td align="right"><sup><sub>56.9</sub></sup></td>
<td align="right"><sup><sub>0.174&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>36.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857281</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.2</sub></sup></td>
<td align="right"><sup><sub>0.544</sub></sup></td>
<td align="right"><sup><sub>13.6</sub></sup></td>
<td align="right"><sup><sub>0.093&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>36.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857345</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.2</sub></sup></td>
<td align="right"><sup><sub>0.546</sub></sup></td>
<td align="right"><sup><sub>27.3</sub></sup></td>
<td align="right"><sup><sub>0.092&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>37.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857389</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.9</sub></sup></td>
<td align="right"><sup><sub>0.647</sub></sup></td>
<td align="right"><sup><sub>16.2</sub></sup></td>
<td align="right"><sup><sub>0.120&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>39.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857890</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.9</sub></sup></td>
<td align="right"><sup><sub>0.647</sub></sup></td>
<td align="right"><sup><sub>32.4</sub></sup></td>
<td align="right"><sup><sub>0.119&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>39.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35857952</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.9</sub></sup></td>
<td align="right"><sup><sub>1.057</sub></sup></td>
<td align="right"><sup><sub>52.9</sub></sup></td>
<td align="right"><sup><sub>0.305&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>41.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35858015</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35858015/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml.01_40_54.1xc565DE/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858015/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml.01_40_54.1xc565DE/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.9</sub></sup></td>
<td align="right"><sup><sub>1.055</sub></sup></td>
<td align="right"><sup><sub>105.5</sub></sup></td>
<td align="right"><sup><sub>0.304&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>40.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35858198</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35858198/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml.01_41_46.CX2InaoG/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858198/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml.01_41_46.CX2InaoG/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.0</sub></sup></td>
<td align="right"><sup><sub>0.799</sub></sup></td>
<td align="right"><sup><sub>40.0</sub></sup></td>
<td align="right"><sup><sub>0.233&nbsp;+&nbsp;0.004</sub></sup></td>
<td align="right"><sup><sub>41.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36761737</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Faster</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.0</sub></sup></td>
<td align="right"><sup><sub>0.800</sub></sup></td>
<td align="right"><sup><sub>80.0</sub></sup></td>
<td align="right"><sup><sub>0.233&nbsp;+&nbsp;0.003</sub></sup></td>
<td align="right"><sup><sub>40.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36761786</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36761786/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml.06_33_22.VqFNuxk6/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36761786/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml.06_33_22.VqFNuxk6/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.6</sub></sup></td>
<td align="right"><sup><sub>0.620</sub></sup></td>
<td align="right"><sup><sub>31.0</sub></sup></td>
<td align="right"><sup><sub>0.181&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>35.8</sub></sup></td>
<td align="right"><sup><sub>31.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35858791</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-C4</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>6.6</sub></sup></td>
<td align="right"><sup><sub>0.620</sub></sup></td>
<td align="right"><sup><sub>62.0</sub></sup></td>
<td align="right"><sup><sub>0.182&nbsp;+&nbsp;0.017</sub></sup></td>
<td align="right"><sup><sub>37.8</sub></sup></td>
<td align="right"><sup><sub>32.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35858828</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.6</sub></sup></td>
<td align="right"><sup><sub>0.889</sub></sup></td>
<td align="right"><sup><sub>22.2</sub></sup></td>
<td align="right"><sup><sub>0.099&nbsp;+&nbsp;0.019</sub></sup></td>
<td align="right"><sup><sub>37.7</sub></sup></td>
<td align="right"><sup><sub>33.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35858933</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.6</sub></sup></td>
<td align="right"><sup><sub>0.897</sub></sup></td>
<td align="right"><sup><sub>44.9</sub></sup></td>
<td align="right"><sup><sub>0.099&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>38.6</sub></sup></td>
<td align="right"><sup><sub>34.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35859007</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>10.2</sub></sup></td>
<td align="right"><sup><sub>1.008</sub></sup></td>
<td align="right"><sup><sub>25.2</sub></sup></td>
<td align="right"><sup><sub>0.126&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>40.0</sub></sup></td>
<td align="right"><sup><sub>35.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35861795</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>10.2</sub></sup></td>
<td align="right"><sup><sub>0.993</sub></sup></td>
<td align="right"><sup><sub>49.7</sub></sup></td>
<td align="right"><sup><sub>0.126&nbsp;+&nbsp;0.017</sub></sup></td>
<td align="right"><sup><sub>40.9</sub></sup></td>
<td align="right"><sup><sub>36.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35861858</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.6</sub></sup></td>
<td align="right"><sup><sub>1.217</sub></sup></td>
<td align="right"><sup><sub>60.9</sub></sup></td>
<td align="right"><sup><sub>0.309&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>42.4</sub></sup></td>
<td align="right"><sup><sub>37.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36494496</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.6</sub></sup></td>
<td align="right"><sup><sub>1.210</sub></sup></td>
<td align="right"><sup><sub>121.0</sub></sup></td>
<td align="right"><sup><sub>0.309&nbsp;+&nbsp;0.015</sub></sup></td>
<td align="right"><sup><sub>42.2</sub></sup></td>
<td align="right"><sup><sub>37.2</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>35859745</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.961</sub></sup></td>
<td align="right"><sup><sub>48.1</sub></sup></td>
<td align="right"><sup><sub>0.239&nbsp;+&nbsp;0.019</sub></sup></td>
<td align="right"><sup><sub>42.1</sub></sup></td>
<td align="right"><sup><sub>37.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36761843</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.975</sub></sup></td>
<td align="right"><sup><sub>97.5</sub></sup></td>
<td align="right"><sup><sub>0.240&nbsp;+&nbsp;0.016</sub></sup></td>
<td align="right"><sup><sub>41.7</sub></sup></td>
<td align="right"><sup><sub>36.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36762092</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36762092/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_2x.yaml.06_37_59.DM5gJYRF/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36762092/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_2x.yaml.06_37_59.DM5gJYRF/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36762092/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_2x.yaml.06_37_59.DM5gJYRF/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
</tr>
<!-- END E2E FASTER AND MASK TABLE -->
</tbody></table>

**Notes:**

- For these models, RPN and the detector are trained jointly and end-to-end.
- Inference time is fully image-to-detections, *including* proposal generation.


### RetinaNet Baselines

<table><tbody>
<!-- START RETINANET TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop.<br/>AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.8</sub></sup></td>
<td align="right"><sup><sub>0.483</sub></sup></td>
<td align="right"><sup><sub>12.1</sub></sup></td>
<td align="right"><sup><sub>0.125</sub></sup></td>
<td align="right"><sup><sub>35.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768636</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768636/12_2017_baselines/retinanet_R-50-FPN_1x.yaml.08_29_48.t4zc9clc/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768636/12_2017_baselines/retinanet_R-50-FPN_1x.yaml.08_29_48.t4zc9clc/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.8</sub></sup></td>
<td align="right"><sup><sub>0.482</sub></sup></td>
<td align="right"><sup><sub>24.1</sub></sup></td>
<td align="right"><sup><sub>0.127</sub></sup></td>
<td align="right"><sup><sub>35.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768677</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.7</sub></sup></td>
<td align="right"><sup><sub>0.666</sub></sup></td>
<td align="right"><sup><sub>16.7</sub></sup></td>
<td align="right"><sup><sub>0.156</sub></sup></td>
<td align="right"><sup><sub>37.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768744</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768744/12_2017_baselines/retinanet_R-101-FPN_1x.yaml.08_31_38.5poQe1ZB/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768744/12_2017_baselines/retinanet_R-101-FPN_1x.yaml.08_31_38.5poQe1ZB/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.7</sub></sup></td>
<td align="right"><sup><sub>0.666</sub></sup></td>
<td align="right"><sup><sub>33.3</sub></sup></td>
<td align="right"><sup><sub>0.154</sub></sup></td>
<td align="right"><sup><sub>37.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768840</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768840/12_2017_baselines/retinanet_R-101-FPN_2x.yaml.08_33_29.grtM0RTf/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768840/12_2017_baselines/retinanet_R-101-FPN_2x.yaml.08_33_29.grtM0RTf/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.6</sub></sup></td>
<td align="right"><sup><sub>1.613</sub></sup></td>
<td align="right"><sup><sub>40.3</sub></sup></td>
<td align="right"><sup><sub>0.341</sub></sup></td>
<td align="right"><sup><sub>39.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768875</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768875/12_2017_baselines/retinanet_X-101-64x4d-FPN_1x.yaml.08_34_37.FSXgMpzP/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768875/12_2017_baselines/retinanet_X-101-64x4d-FPN_1x.yaml.08_34_37.FSXgMpzP/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.6</sub></sup></td>
<td align="right"><sup><sub>1.625</sub></sup></td>
<td align="right"><sup><sub>81.3</sub></sup></td>
<td align="right"><sup><sub>0.339</sub></sup></td>
<td align="right"><sup><sub>39.2</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36768907</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36768907/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml.08_35_40.pF3nzPpu/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36768907/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml.08_35_40.pF3nzPpu/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.7</sub></sup></td>
<td align="right"><sup><sub>1.343</sub></sup></td>
<td align="right"><sup><sub>33.6</sub></sup></td>
<td align="right"><sup><sub>0.277</sub></sup></td>
<td align="right"><sup><sub>39.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36769563</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36769563/12_2017_baselines/retinanet_X-101-32x8d-FPN_1x.yaml.08_42_05.06JTK6vJ/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36769563/12_2017_baselines/retinanet_X-101-32x8d-FPN_1x.yaml.08_42_05.06JTK6vJ/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>RetinaNet</sub></sup></td>
<td align="left"><sup><sub>2x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.7</sub></sup></td>
<td align="right"><sup><sub>1.340</sub></sup></td>
<td align="right"><sup><sub>67.0</sub></sup></td>
<td align="right"><sup><sub>0.276</sub></sup></td>
<td align="right"><sup><sub>38.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>36769641</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36769641/12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml.08_42_55.sUPnwXI5/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36769641/12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml.08_42_55.sUPnwXI5/output/test/coco_2014_minival/retinanet/detections_coco_2014_minival_results.json">boxes</a></sub></sup></td>
</tr>
<!-- END RETINANET TABLE -->
</tbody></table>

**Notes:** none

### Mask R-CNN with Bells & Whistles

<table><tbody>
<!-- START BELLS TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp<br/>AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop.<br/>AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>X-152-32x8d-FPN-IN5k</sub></sup></td>
<td align="left"><sup><sub>Mask</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>1</sub></sup></td>
<td align="right"><sup><sub>9.6</sub></sup></td>
<td align="right"><sup><sub>1.188</sub></sup></td>
<td align="right"><sup><sub>85.8</sub></sup></td>
<td align="right"><sup><sub>12.100&nbsp;+&nbsp;0.046</sub></sup></td>
<td align="right"><sup><sub>48.1</sub></sup></td>
<td align="right"><sup><sub>41.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37129812</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37129812/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml.09_35_36.8pzTQKYK/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37129812/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml.09_35_36.8pzTQKYK/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37129812/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml.09_35_36.8pzTQKYK/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json">masks</a></sub></sup></td>
<tr>
<td align="left"><sup><sub>[above without test-time aug.]</sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub>0.325&nbsp;+&nbsp;0.018</sub></sup></td>
<td align="right"><sup><sub>45.2</sub></sup></td>
<td align="right"><sup><sub>39.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
<td align="right"><sup><sub></sub></sup></td>
</tr>
<!-- END BELLS TABLE -->
</tbody></table>

**Notes:**

- A deeper backbone architecture is used: ResNeXt-**152**-32x8d-FPN
- The backbone ResNeXt-152-32x8d model was trained on ImageNet-**5k** (not the usual ImageNet-1k)
- Training uses multi-scale jitter over scales {640, 672, 704, 736, 768, 800}
- Row 1: test-time augmentations are multi-scale testing over {400, 500, 600, 700, 900, 1000, 1100, 1200} and horizontal flipping (on each scale)
- Row 2: same model as row 1, but without any test-time augmentation (i.e., same as the common baseline configuration)
- Like the other results, this is a single model result (it is not an ensemble of models)

## Keypoint Detection Baselines

#### Common Settings for Keypoint Detection Baselines (That Differ from Boxes and Masks)

Our keypoint detection baselines differ from our box and mask baselines in a couple of details:

- Due to less training data for the keypoint detection task compared with boxes and masks, we enable multi-scale jitter during training for all keypoint detection models. (Testing is still without any test-time augmentations by default.)
- Models are trained only on images from `coco_2014_train` union `coco_2014_valminusminival` that contain at least one person with keypoint annotations (all other images are discarded from the training set).
- Metrics are reported for the person class only (still run on the entire `coco_2014_minival` dataset).

### Person-Specific RPN Baselines

<table><tbody>
<!-- START PERSON-ONLY RPN TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop. AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>6.4</sub></sup></td>
<td align="right"><sup><sub>0.391</sub></sup></td>
<td align="right"><sup><sub>9.8</sub></sup></td>
<td align="right"><sup><sub>0.082</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>64.0</sub></sup></td>
<td align="right"><sup><sub>35998996</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35998996/12_2017_baselines/rpn_person_only_R-50-FPN_1x.yaml.08_10_08.0ZWmJm6F/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998996/12_2017_baselines/rpn_person_only_R-50-FPN_1x.yaml.08_10_08.0ZWmJm6F/output/test/keypoints_coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998996/12_2017_baselines/rpn_person_only_R-50-FPN_1x.yaml.08_10_08.0ZWmJm6F/output/test/keypoints_coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35998996/12_2017_baselines/rpn_person_only_R-50-FPN_1x.yaml.08_10_08.0ZWmJm6F/output/test/keypoints_coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>8.1</sub></sup></td>
<td align="right"><sup><sub>0.504</sub></sup></td>
<td align="right"><sup><sub>12.6</sub></sup></td>
<td align="right"><sup><sub>0.109</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.2</sub></sup></td>
<td align="right"><sup><sub>35999521</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35999521/12_2017_baselines/rpn_person_only_R-101-FPN_1x.yaml.08_20_33.1OkqMmqP/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999521/12_2017_baselines/rpn_person_only_R-101-FPN_1x.yaml.08_20_33.1OkqMmqP/output/test/keypoints_coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999521/12_2017_baselines/rpn_person_only_R-101-FPN_1x.yaml.08_20_33.1OkqMmqP/output/test/keypoints_coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999521/12_2017_baselines/rpn_person_only_R-101-FPN_1x.yaml.08_20_33.1OkqMmqP/output/test/keypoints_coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>11.5</sub></sup></td>
<td align="right"><sup><sub>1.394</sub></sup></td>
<td align="right"><sup><sub>34.9</sub></sup></td>
<td align="right"><sup><sub>0.289</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.9</sub></sup></td>
<td align="right"><sup><sub>35999553</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/35999553/12_2017_baselines/rpn_person_only_X-101-64x4d-FPN_1x.yaml.08_21_33.ghFzzArr/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999553/12_2017_baselines/rpn_person_only_X-101-64x4d-FPN_1x.yaml.08_21_33.ghFzzArr/output/test/keypoints_coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999553/12_2017_baselines/rpn_person_only_X-101-64x4d-FPN_1x.yaml.08_21_33.ghFzzArr/output/test/keypoints_coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/35999553/12_2017_baselines/rpn_person_only_X-101-64x4d-FPN_1x.yaml.08_21_33.ghFzzArr/output/test/keypoints_coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>RPN</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>11.6</sub></sup></td>
<td align="right"><sup><sub>1.104</sub></sup></td>
<td align="right"><sup><sub>27.6</sub></sup></td>
<td align="right"><sup><sub>0.224</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.2</sub></sup></td>
<td align="right"><sup><sub>36760438</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/36760438/12_2017_baselines/rpn_person_only_X-101-32x8d-FPN_1x.yaml.06_04_23.M2oJlDPW/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;props:&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760438/12_2017_baselines/rpn_person_only_X-101-32x8d-FPN_1x.yaml.06_04_23.M2oJlDPW/output/test/keypoints_coco_2014_train/generalized_rcnn/rpn_proposals.pkl">1</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760438/12_2017_baselines/rpn_person_only_X-101-32x8d-FPN_1x.yaml.06_04_23.M2oJlDPW/output/test/keypoints_coco_2014_valminusminival/generalized_rcnn/rpn_proposals.pkl">2</a>,&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/36760438/12_2017_baselines/rpn_person_only_X-101-32x8d-FPN_1x.yaml.06_04_23.M2oJlDPW/output/test/keypoints_coco_2014_minival/generalized_rcnn/rpn_proposals.pkl">3</a></sub></sup></td>
</tr>
<!-- END PERSON-ONLY RPN TABLE -->
</tbody></table>

**Notes:**

- *Metrics are for the person category only.*
- Inference time only includes RPN proposal generation.
- "prop. AR" is proposal average recall at 1000 proposals per image.
- Proposal download links ("props"): "1" is `coco_2014_train`; "2" is `coco_2014_valminusminival`; and "3" is `coco_2014_minival`. These include all images, not just the ones with valid keypoint annotations.

### Keypoint-Only Mask R-CNN Baselines Using Precomputed RPN Proposals

<table><tbody>
<!-- START 2-STAGE KEYPOINTS TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop. AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.533</sub></sup></td>
<td align="right"><sup><sub>13.3</sub></sup></td>
<td align="right"><sup><sub>0.081&nbsp;+&nbsp;0.087</sub></sup></td>
<td align="right"><sup><sub>52.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>64.1</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37651787</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37651787/12_2017_baselines/keypoint_rcnn_R-50-FPN_1x.yaml.20_00_48.UiwJsTXB/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/gene
ralized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651787/12_2017_baselines/keypoint_rcnn_R-50-FPN_1x.yaml.20_00_48.UiwJsTXB/output/test/keypoints_coco_2014_minival/generalized_rcnn
/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651787/12_2017_baselines/keypoint_rcnn_R-50-FPN_1x.yaml.20_00_48.UiwJsTXB/output/test/keypoints_coco_2014_miniva
l/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>7.7</sub></sup></td>
<td align="right"><sup><sub>0.533</sub></sup></td>
<td align="right"><sup><sub>19.2</sub></sup></td>
<td align="right"><sup><sub>0.080&nbsp;+&nbsp;0.085</sub></sup></td>
<td align="right"><sup><sub>53.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37651887</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37651887/12_2017_baselines/keypoint_rcnn_R-50-FPN_s1x.yaml.20_01_40.FDjUQ7VX/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/gen
eralized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651887/12_2017_baselines/keypoint_rcnn_R-50-FPN_s1x.yaml.20_01_40.FDjUQ7VX/output/test/keypoints_coco_2014_minival/generalized_rc
nn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651887/12_2017_baselines/keypoint_rcnn_R-50-FPN_s1x.yaml.20_01_40.FDjUQ7VX/output/test/keypoints_coco_2014_min
ival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.4</sub></sup></td>
<td align="right"><sup><sub>0.668</sub></sup></td>
<td align="right"><sup><sub>16.7</sub></sup></td>
<td align="right"><sup><sub>0.109&nbsp;+&nbsp;0.080</sub></sup></td>
<td align="right"><sup><sub>53.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37651996</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37651996/12_2017_baselines/keypoint_rcnn_R-101-FPN_1x.yaml.20_02_37.eVXnKM2Q/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/gen
eralized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651996/12_2017_baselines/keypoint_rcnn_R-101-FPN_1x.yaml.20_02_37.eVXnKM2Q/output/test/keypoints_coco_2014_minival/generalized_rc
nn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37651996/12_2017_baselines/keypoint_rcnn_R-101-FPN_1x.yaml.20_02_37.eVXnKM2Q/output/test/keypoints_coco_2014_min
ival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.4</sub></sup></td>
<td align="right"><sup><sub>0.668</sub></sup></td>
<td align="right"><sup><sub>24.1</sub></sup></td>
<td align="right"><sup><sub>0.108&nbsp;+&nbsp;0.076</sub></sup></td>
<td align="right"><sup><sub>54.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37652016</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37652016/12_2017_baselines/keypoint_rcnn_R-101-FPN_s1x.yaml.20_03_32.z86wT97d/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/ge
neralized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37652016/12_2017_baselines/keypoint_rcnn_R-101-FPN_s1x.yaml.20_03_32.z86wT97d/output/test/keypoints_coco_2014_minival/generalized_
rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37652016/12_2017_baselines/keypoint_rcnn_R-101-FPN_s1x.yaml.20_03_32.z86wT97d/output/test/keypoints_coco_2014_
minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.8</sub></sup></td>
<td align="right"><sup><sub>1.477</sub></sup></td>
<td align="right"><sup><sub>36.9</sub></sup></td>
<td align="right"><sup><sub>0.288&nbsp;+&nbsp;0.077</sub></sup></td>
<td align="right"><sup><sub>55.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.7</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37731079</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37731079/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_40_56.wj7Hg7lX/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminiv
al/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731079/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_40_56.wj7Hg7lX/output/test/keypoints_coco_2014_minival/ge
neralized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731079/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_40_56.wj7Hg7lX/output/test/keypo
ints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.9</sub></sup></td>
<td align="right"><sup><sub>1.478</sub></sup></td>
<td align="right"><sup><sub>53.4</sub></sup></td>
<td align="right"><sup><sub>0.286&nbsp;+&nbsp;0.075</sub></sup></td>
<td align="right"><sup><sub>56.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>67.1</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37731142</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37731142/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_41_54.e1sD4Frh/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusmini
val/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731142/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_41_54.e1sD4Frh/output/test/keypoints_coco_2014_minival/
generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731142/12_2017_baselines/keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_41_54.e1sD4Frh/output/test/ke
ypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.9</sub></sup></td>
<td align="right"><sup><sub>1.215</sub></sup></td>
<td align="right"><sup><sub>30.4</sub></sup></td>
<td align="right"><sup><sub>0.219&nbsp;+&nbsp;0.084</sub></sup></td>
<td align="right"><sup><sub>55.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.2</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37730253</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37730253/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_34_24.3G9OcQuR/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminiv
al/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37730253/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_34_24.3G9OcQuR/output/test/keypoints_coco_2014_minival/ge
neralized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37730253/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_34_24.3G9OcQuR/output/test/keypo
ints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>12.9</sub></sup></td>
<td align="right"><sup><sub>1.214</sub></sup></td>
<td align="right"><sup><sub>43.8</sub></sup></td>
<td align="right"><sup><sub>0.218&nbsp;+&nbsp;0.071</sub></sup></td>
<td align="right"><sup><sub>55.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>67.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37731010</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37731010/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_39_51.xt1oMzRk/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusmini
val/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731010/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_39_51.xt1oMzRk/output/test/keypoints_coco_2014_minival/
generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37731010/12_2017_baselines/keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_39_51.xt1oMzRk/output/test/ke
ypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<!-- END 2-STAGE KEYPOINTS TABLE -->
</tbody></table>

**Notes:**

- *Metrics are for the person category only.*
- Each row uses precomputed RPN proposals from the corresponding table row above that uses the same backbone.
- Inference time *excludes* proposal generation.


### End-to-End Keypoint-Only Mask R-CNN Baselines

<table><tbody>
<!-- START END-TO-END KEYPOINTS TABLE -->
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backbone&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>type</sub></sup></th>
<th valign="bottom"><sup><sub>lr<br/>schd</sub></sup></th>
<th valign="bottom"><sup><sub>im/<br/>gpu</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>mem<br/>(GB)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>(s/iter)</sub></sup></th>
<th valign="bottom"><sup><sub>train<br/>time<br/>total<br/>(hr)</sub></sup></th>
<th valign="bottom"><sup><sub>inference<br/>time<br/>(s/im)</sub></sup></th>
<th valign="bottom"><sup><sub>box AP</sub></sup></th>
<th valign="bottom"><sup><sub>mask AP</sub></sup></th>
<th valign="bottom"><sup><sub>kp AP</sub></sup></th>
<th valign="bottom"><sup><sub>prop. AR</sub></sup></th>
<th valign="bottom"><sup><sub>model id</sub></sup></th>
<th valign="bottom"><sup><sub>download<br/>links</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.0</sub></sup></td>
<td align="right"><sup><sub>0.832</sub></sup></td>
<td align="right"><sup><sub>20.8</sub></sup></td>
<td align="right"><sup><sub>0.097&nbsp;+&nbsp;0.092</sub></sup></td>
<td align="right"><sup><sub>53.6</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>64.2</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37697547</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-50-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>9.0</sub></sup></td>
<td align="right"><sup><sub>0.828</sub></sup></td>
<td align="right"><sup><sub>29.9</sub></sup></td>
<td align="right"><sup><sub>0.096&nbsp;+&nbsp;0.089</sub></sup></td>
<td align="right"><sup><sub>54.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.4</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37697714</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>10.6</sub></sup></td>
<td align="right"><sup><sub>0.923</sub></sup></td>
<td align="right"><sup><sub>23.1</sub></sup></td>
<td align="right"><sup><sub>0.124&nbsp;+&nbsp;0.084</sub></sup></td>
<td align="right"><sup><sub>54.5</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>64.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37697946</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37697946/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_1x.yaml.08_45_06.Y14KqbST/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697946/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_1x.yaml.08_45_06.Y14KqbST/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37697946/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_1x.yaml.08_45_06.Y14KqbST/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>R-101-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>10.6</sub></sup></td>
<td align="right"><sup><sub>0.921</sub></sup></td>
<td align="right"><sup><sub>33.3</sub></sup></td>
<td align="right"><sup><sub>0.123&nbsp;+&nbsp;0.083</sub></sup></td>
<td align="right"><sup><sub>55.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>65.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37698009</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>14.1</sub></sup></td>
<td align="right"><sup><sub>1.655</sub></sup></td>
<td align="right"><sup><sub>41.4</sub></sup></td>
<td align="right"><sup><sub>0.302&nbsp;+&nbsp;0.079</sub></sup></td>
<td align="right"><sup><sub>56.3</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37732355</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37732355/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_56_16.yv4t4W8N/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732355/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_56_16.yv4t4W8N/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732355/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_56_16.yv4t4W8N/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-64x4d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>14.1</sub></sup></td>
<td align="right"><sup><sub>1.731</sub></sup></td>
<td align="right"><sup><sub>62.5</sub></sup></td>
<td align="right"><sup><sub>0.322&nbsp;+&nbsp;0.074</sub></sup></td>
<td align="right"><sup><sub>56.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.8</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37732415</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37732415/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_57_48.Spqtq3Sf/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732415/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_57_48.Spqtq3Sf/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732415/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_57_48.Spqtq3Sf/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>14.2</sub></sup></td>
<td align="right"><sup><sub>1.410</sub></sup></td>
<td align="right"><sup><sub>35.3</sub></sup></td>
<td align="right"><sup><sub>0.235&nbsp;+&nbsp;0.080</sub></sup></td>
<td align="right"><sup><sub>56.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>66.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37792158</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37792158/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_54_16.LgZeo40k/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37792158/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_54_16.LgZeo40k/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37792158/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_54_16.LgZeo40k/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<tr>
<td align="left"><sup><sub>X-101-32x8d-FPN</sub></sup></td>
<td align="left"><sup><sub>Kps</sub></sup></td>
<td align="left"><sup><sub>s1x</sub></sup></td>
<td align="right"><sup><sub>2</sub></sup></td>
<td align="right"><sup><sub>14.2</sub></sup></td>
<td align="right"><sup><sub>1.408</sub></sup></td>
<td align="right"><sup><sub>50.8</sub></sup></td>
<td align="right"><sup><sub>0.236&nbsp;+&nbsp;0.075</sub></sup></td>
<td align="right"><sup><sub>56.9</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>67.0</sub></sup></td>
<td align="right"><sup><sub>-</sub></sup></td>
<td align="right"><sup><sub>37732318</sub></sup></td>
<td align="left"><sup><sub><a href="https://dl.fbaipublicfiles.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/test/keypoints_coco_2014_minival/generalized_rcnn/bbox_keypoints_coco_2014_minival_results.json">boxes</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/test/keypoints_coco_2014_minival/generalized_rcnn/keypoints_keypoints_coco_2014_minival_results.json">kps</a></sub></sup></td>
</tr>
<!-- END END-TO-END KEYPOINTS TABLE -->
</tbody></table>

**Notes:**

- *Metrics are for the person category only.*
- For these models, RPN and the detector are trained jointly and end-to-end.
- Inference time is fully image-to-detections, *including* proposal generation.
