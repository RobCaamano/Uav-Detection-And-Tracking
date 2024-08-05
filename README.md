# Uav-Detection-And-Tracking
<a href="https://github.com/tensorflow/models/tree/master/research/object_detection" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Object Detection"/></a>
<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Model Zoo"/></a>

## Sections

- [About](#about)
- [Files](#files)
- [Demo Videos](#demo-kalman)
- [Demo Images](#demo)
- [Training Metrics](#metrics)

## About <a id="about"></a>

Utlizes TensorFlow2 Object Detection API with a finetune of TensorFlow's faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 model.

## Files <a id="files"></a>

UAV_FasterRCNN.py: Used for inference

XML_To_TFRecord.py: Converts XML files into training and validation TFRecord files

download_model.py: Used to download base model

model_main_tf2.py: Slightly modified TF2 Training Loop

vids_to_frames: Extracts frames from video and saves them in a given directory

kalman_filter.py: Implements Kalman filter to track drone's trajectory. Compiles detections into a video with a line tracking the drone.


## Demo Detections Videos with Kalman Filter <a id="demo-kalman"></a>

Both videos are available in this repo's [/kf_vids](https://github.com/RobCaamano/Uav-Detection-And-Tracking/tree/main/kf_vids) directory.

### Demo #1

https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/eaf5e702-f3a6-4e5b-b7d6-4c06a39e6c6a


### Demo # 2

https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/81024abb-f7ec-49d2-a071-c2120a5a9ea3


## Demo Detections Images <a id="demo"></a>

More samples are available in this repo's [./detections](https://github.com/RobCaamano/Uav-Detection-And-Tracking/tree/main/detections) directory.

### Demo #1

![frame 0](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_0.jpg)

### Demo #2

![frame 7547](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_7547.jpg)


## Training Metrics <a id="metrics"></a>

![Metrics 1](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/metrics_1.png)

![Learning rate](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/lr.png)
