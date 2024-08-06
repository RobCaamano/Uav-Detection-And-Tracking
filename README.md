# Uav Detection And Tracking
<a href="https://github.com/tensorflow/models/tree/master/research/object_detection" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Object Detection"/></a>
<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Model Zoo"/></a>

## Sections

- [About](#about)
- [Files](#files)
- [Demo Videos](#demo-kalman)
- [Demo Images](#demo)
- [Training Metrics](#metrics)

## About <a id="about"></a>

The UAV Detection and Tracking project aims to identify and monitor unmanned aerial vehicles (UAVs) using advanced object detection and tracking techniques. It utlizes TensorFlow2 Object Detection API with a finetune of TensorFlow's '[faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.config)' model.

- Fine-Tuned Object Detection Model: Utilizes a fine-tuned version of the TensorFlow Faster R-CNN ResNet101 model, optimized for detecting UAVs

- End-to-End Pipeline: Includes tools for model preparation, video frame extraction, data conversion, and tracking.

- Tracking and Visualization: Implements a Kalman filter for trajectory tracking and generates visualizations to show the UAV’s path.

## Files <a id="files"></a>

- **download_model.py:** Used to fetch the pre-trained model from the TensorFlow model zoo.

- **vids_to_frames:** Extracts frames from video and saves them in a given directory using OpenCV.

- **XML_To_TFRecord.py:** Converts XML annotation files into TensorFlow’s TFRecord format, which is required for training and validation of the detection model.

- **model_main_tf2.py:** Contains a slightly modified TensorFlow 2 training loop tailored for fine-tuning the object detection model. Manages the training process, including model checkpointing and evaluation.

- **UAV_FasterRCNN.py:** Used for inference with the fine-tuned Faster R-CNN model. Performs object detection on new video frames, identifying UAVs and generating bounding boxes around detected objects.

- **kalman_filter.py:** Implements the Kalman filter algorithm to track the trajectory of detected UAVs. Processes the detection results and compiles them into a video with a line tracing the UAV’s movement, providing a visualization of its path.


## Demo Videos with Kalman Filter <a id="demo-kalman"></a>

Both videos are available in this repo's [/kf_vids](https://github.com/RobCaamano/Uav-Detection-And-Tracking/tree/main/kf_vids) directory.

### Demo #1

https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/eaf5e702-f3a6-4e5b-b7d6-4c06a39e6c6a

### Demo # 2

https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/81024abb-f7ec-49d2-a071-c2120a5a9ea3


## Demo Images <a id="demo"></a>

More samples are available in this repo's [./detections](https://github.com/RobCaamano/Uav-Detection-And-Tracking/tree/main/detections) directory.

### Demo #1

![frame 0](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_0.jpg)

### Demo #2

![frame 7547](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_7547.jpg)


## Training Metrics <a id="metrics"></a>

![Metrics 1](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/metrics_1.png)

![Learning rate](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/lr.png)
