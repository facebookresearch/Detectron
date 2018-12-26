# FAQ

This document covers frequently asked questions.

- For general information about Detectron, please see [`README.md`](README.md).
- For installation instructions, please see [`INSTALL.md`](INSTALL.md).
- For a quick getting started guide, please see [`GETTING_STARTED.md`](GETTING_STARTED.md).

#### Q: How do I compute validation AP during training?

**A:** Detectron does not compute validation statistics (e.g., AP) during training because this slows training. Instead, we've implemented a "validation monitor", which is a process that polls for new model checkpoints saved by a training job and when one is found performs inference with it by scheduling a job with `tools/test_net.py` asynchronously using free GPUs in our cluster. We have not released the validation monitor because (1) it's a relatively thin wrapper on top of `tools/train_net.py` and (2) the little code that comprises it is specific to our cluster and would not be generally useful.

#### Q: How do I restrict Detectron to use only a subset of the GPUs on a server?

**A:** Don't modify the code; use the [`CUDA_VISIBLE_DEVICES`](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) environment variable instead.

#### Q: Detection on one image is really slow compared to the reported performance, why?

A: Various algorithms and caches (e.g., from `cudnn`) take some time to warm up. Peak inference performance will not be reached until after a few images have been processed.

Also potentially relevant: inference with Mask R-CNN on high-resolution images may be slow simply because substantial time is spent upsampling the predicted masks to the original image resolution (this has not been optimized). You can diagnose this issue if the `misc_mask` time reported by `tools/infer_simple.py` is high (e.g., much more than 20-90ms). The solution is to first resize your images such that the short side is around 600-800px (the exact choice does not matter) and then run inference on the resized image.


#### Q: How do I implement a custom Caffe2 CPU or GPU operator for use in Detectron?

**A:** Detectron uses a number of specialized Caffe2 operators that are distributed via the [Caffe2 Detectron module](https://github.com/pytorch/pytorch/tree/master/modules/detectron) as part of the core Caffe2 GitHub repository. If you'd like to implement a custom Caffe2 operator for your project, we have written a toy example illustrating how to add an operator under the Detectron source tree; please see [`detectron/ops/zero_even_op.*`](detectron/ops/) and [`detectron/tests/test_zero_even_op.py`](detectron/tests/test_zero_even_op.py). For more background on writing Caffe2 operators please consult the [Caffe2 documentation](https://caffe2.ai/docs/custom-operators.html).

#### Q: How do I use Detectron to train a model on a custom dataset?

**A:** If possible, we strongly recommend that you first convert the custom dataset annotation format to the [COCO API json format](http://cocodataset.org/#download). Then, add your dataset to the [dataset catalog](detectron/datasets/dataset_catalog.py) so that Detectron can use it for training and inference. If your dataset cannot be converted to the COCO API json format, then it's likely that more significant code modifications will be required. If the dataset you're adding is popular, please consider making the converted annotations publicly available; If code modifications are required, please consider submitting a pull request.
