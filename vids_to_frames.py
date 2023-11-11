import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Extracting frames from {os.path.basename(video_path)}", unit='frame') as pbar:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
            pbar.update(1)
    cap.release()

# Train Data
#train_videos = os.listdir('./train_vids')
#for video_name in train_videos:
#    video_path = os.path.join('./train_vids', video_name)
#    output_folder = os.path.join('./train_frames', video_name.split('.')[0]) 
#    extract_frames(video_path, output_folder)

# Test Data
test_videos = os.listdir('./test_vids')
for video_name in test_videos:
    video_path = os.path.join('./test_vids', video_name)
    output_folder = os.path.join('./test_frames', video_name.split('.')[0]) 
    extract_frames(video_path, output_folder)