# Running Detectron via Docker

The current Docker repo for Caffe2 (caffe2:cuda8-cudnn6-all-options) doesn't build from scratch, because of a broken dependency downstream.

There's a modified Dockerfile to build the required Caffe2 image in `Docker-Caffe2`.

Create the image with the following commands:

```bash
$ cd docker/Docker-Caffe2
$ docker build -t caffe2:cuda8-cudnn6-all-options .
```

You should then be able to build the included Dockerfile in this repo, and run it using nvidia-docker.

```bash
$ cd ..
$ docker build -t detectron .
```
