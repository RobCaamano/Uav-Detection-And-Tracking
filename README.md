# Uav-Detection-And-Tracking
Utlizes TensorFlow2 Object Detection API with a finetune of TensorFlow's faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 model.
## Training Metrics
![Metrics 1](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/metrics_1.png)

![Learning rate](https://github.com/RobCaamano/Uav-Detection-And-Tracking/blob/main/lr.png)

## Detections from Test Videos
In /detections directory

## File Explanations:
UAV_FasterRCNN.py: Used for inference
XML_To_TFRecord.py: Converts XML files into training and validation TFRecord files
download_model.py: Used to download base model
model_main_tf2.py: Slightly modified TF2 Training Loop
vids_to_frames: Extracts frames from video and saves them in a given directory
