from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import cv2
import os
from tqdm import tqdm
import json

# Set up Kalman Filter
def kalman_filter(initial_position=None, frame_size=(1280, 720), dt=1/20.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State Transition Matrix
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Measurement Matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Initial State
    if initial_position is None:
        kf.x = np.array([0., 0., 0., 0.])
    else:
        kf.x = np.array([initial_position[0] * frame_size[0], initial_position[1] * frame_size[1], 0., 0.])

    # Process Noise
    kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)

    # Measurement Noise
    kf.R = np.array([[1, 0], [0, 1]]) * 0.05

    # Initial Uncertainty
    kf.P *= 5

    return kf

# Kalman Filter Update
def kalman_update(kf, measurement, frame_size=(1280, 720)):
    scaled_measurement = np.array([measurement[0] * frame_size[0], measurement[1] * frame_size[1]])
    kf.predict()
    kf.update(scaled_measurement)

    return kf.x

# Process Detections and Create Video
def process_detections(detections_dir, detected_positions, output_video, frame_size=(1280, 720)):
    images = sorted([img for img in os.listdir(detections_dir) if img.endswith(".jpg") or img.endswith(".png")], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, frame_size)

    initial_position = detected_positions[0] if detected_positions else None

    kf = kalman_filter(initial_position, frame_size)
    trajectory = []
    prev_frame_num = -1

    for idx, img in tqdm(enumerate(images), desc=f"Processing {detections_dir}"):
        # Extract frame number from filename
        frame_number = int(img.split('_')[-1].split('.')[0])
        frame_jump = frame_number - prev_frame_num > 1

        if frame_jump or kf is None:
            initial_position = detected_positions[idx] if idx < len(detected_positions) else None
            kf = kalman_filter(initial_position, frame_size)

        frame = cv2.imread(os.path.join(detections_dir, img))
        if frame is None:
            continue

        if idx < len(detected_positions):
            position = detected_positions[idx]
            kf_x, kf_y, _, _ = kalman_update(kf, position, frame_size)
        else:
            kf.predict()
            kf_x, kf_y = kf.x[0], kf.x[1]

        trajectory.append((int(kf_x), int(kf_y)))

        frame_with_trajectory = frame.copy()
        for i in range(1, len(trajectory)):
            cv2.line(frame_with_trajectory, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

        out.write(cv2.resize(frame_with_trajectory, frame_size))
        prev_frame_num = frame_number

    out.release()

# Set up paths
directories = ["./detections/Drone Tracking 1", "./detections/Drone Tracking 2"]
outputs = ["./kf_vids/output1.avi", "./kf_vids/output2.avi"]
detected_positions_paths = ["./detected_positions_Drone Tracking 1.json", "./detected_positions_Drone Tracking 2.json"]

for dir, out, dp_file in zip(directories, outputs, detected_positions_paths):
    with open(dp_file, 'r') as f:
        detected_positions = json.load(f)
    process_detections(dir, detected_positions, out)