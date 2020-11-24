import os
import csv
import glob
import cv2
from tqdm import tqdm


def duration(video_dir, duration_dir):
    split_names = ["train", "test", "valid"]
    with open(os.path.join(duration_dir, "duration_frame.csv"), 'w') as f:
        for split in split_names:
            video_dir_split = os.path.join(video_dir, split)
            videos = sorted(glob.glob(video_dir_split + '/*' * 2 + '.mp4'))

            for video in tqdm(videos):
                capture = cv2.VideoCapture(video)
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                video_name = os.path.basename(video)
                writer = csv.writer(f)
                writer.writerow([video_name[:-4], duration, frame_count])
