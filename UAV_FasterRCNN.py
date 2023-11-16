import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import os
from tqdm import tqdm
import numpy as np
import json

# Load the trained Faster RCNN model
config_path = 'C:/Users/caama/.keras/datasets/faster_rcnn_resnet101_v1_800x1333_uav/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(config_path)
model_config = configs['model']

detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore the checkpoint
ckpt_path = './updated_model'
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).expect_partial()

tf.saved_model.save(detection_model, './updated_model/saved_model')

# Load the saved model
#detection_model = tf.saved_model.load('./updated_model/saved_model')

# Dummy image
detection_model(tf.zeros([1, 1024, 1024, 3]))

# Run model inference
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

# Load the label map
label_map_path = './label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the test frames
test_frames_path = ['./test_frames/Drone Tracking 1', './test_frames/Drone Tracking 2']

# Create detection dir if it doesn't exist
detection_path = './detections'
if not os.path.exists(detection_path):
    os.makedirs(detection_path)

# Set the confidence threshold
threshold = 0.5

# Loop through the test frames
for path in test_frames_path:
    detection_sub_path = os.path.join(detection_path, os.path.basename(path))
    if not os.path.exists(detection_sub_path):
        os.makedirs(detection_sub_path)
    
    frames = os.listdir(path)
    frames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    detected_positions = []

    for frame in tqdm(frames, desc=f"Processing frames in {path}"):
        frame_path = os.path.join(path, frame)
        img = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        # Convert img to float32 and normalize
        input_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Detect objects in the frame
        detections = detect_fn(input_tensor)

        # Select the detection with the highest score
        scores = detections['detection_scores'].numpy()[0]
        boxes = detections['detection_boxes'].numpy()[0]
        max_score_index = np.argmax(scores)
        max_score = scores[max_score_index]

        if max_score > threshold:
            box = boxes[max_score_index]
            ymin, xmin, ymax, xmax = box

            # Standardize positions to image size
            x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
            detected_positions.append((x_center, y_center))

            # Visualize the detection with the highest score
            viz_utils.visualize_boxes_and_labels_on_image_array(
                img_rgb,
                np.array([detections['detection_boxes'].numpy()[0][max_score_index]]),
                np.array([detections['detection_classes'].numpy()[0][max_score_index].astype(int)]),
                np.array([max_score]),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                skip_labels=True)

            # Save the frame with visualization
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            detected_frame_path = os.path.join(detection_sub_path, frame)
            cv2.imwrite(detected_frame_path, img_bgr)

    # Save detected positions to JSON file for path
    json_path = f'./detected_positions_{os.path.basename(path)}.json'
    with open(json_path, 'w') as f:
        json.dump(detected_positions, f)
    print(f"Saved detected positions to '{json_path}'")

print("Detections completed")