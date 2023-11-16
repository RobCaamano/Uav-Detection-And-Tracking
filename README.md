# Uav-Detection-And-Tracking
<a href="https://github.com/tensorflow/models/tree/master/research/object_detection" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Object Detection"/></a>
<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md" target="_parent"><img src="https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow" alt="TF2 Model Zoo"/></a>

Utlizes TensorFlow2 Object Detection API with a finetune of TensorFlow's faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 model.

## File Explanations:
UAV_FasterRCNN.py: Used for inference

XML_To_TFRecord.py: Converts XML files into training and validation TFRecord files

download_model.py: Used to download base model

model_main_tf2.py: Slightly modified TF2 Training Loop

vids_to_frames: Extracts frames from video and saves them in a given directory

kalman_filter.py: Implements Kalman filter to track drone's trajectory. Compiles detections into a video with a line tracking the drone.

## Detections from Test Videos
Samples in /detections directory. There are too many to fit.

### Drone Tracking 1 Example Detection
![frame 0](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_0.jpg)

### Drone Tracking 2 Example Detection
![frame 7547](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/detections/frame_7547.jpg)

## Detections Video with Kalman Filter
Present in /kf_vids directory

### Drone Tracking 1


https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/73fa7541-8289-4073-93a1-1cc7213ae03c



### Drone Tracking 2


https://github.com/RobCaamano/Uav-Detection-And-Tracking/assets/65639885/81024abb-f7ec-49d2-a071-c2120a5a9ea3



## Training Metrics
![Metrics 1](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/metrics_1.png)

![Learning rate](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/lr.png)
