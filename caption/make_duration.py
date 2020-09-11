import os
import csv

import cv2
from tqdm import tqdm

"""
for making correspondence between duration and frame
"""

data_list = "./tmp/divided_videos/2020-09-03/"
videos = sorted(os.listdir(data_list))
for video in tqdm(videos):
    video_path = os.path.join(data_list, video)
    video_elements = sorted(os.listdir(video_path))
    for element in video_elements:
        video_element_path = os.path.join(video_path, element)
        capture = cv2.VideoCapture(video_element_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        with open("./duration_frame.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([element, duration, frame_count])