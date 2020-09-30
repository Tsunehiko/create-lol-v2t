import os
import json

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

"""
get information (duration, num of sentences, num of words, etc) about the dataset.
"""

dataset_name = "testing"
video_dir = "../dataset/testing"
annotation_path = "/home/Tanaka/densecap/data/lol/annotation/testing.json"

print("===== video =====")
videos = sorted(glob.glob(video_dir + '/*' * 2 + '.mp4'))
num_videos = 0
all_duration = 0
durations = []

num_videos = len(videos)
for video in tqdm(videos):
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    durations.append(duration)
    all_duration += duration

print("===== annotation =====")
num_sentences = 0
num_words = 0

with open(annotation_path, 'r') as data_file:
    data = json.load(data_file)

sent_durations = []
for vid, val in tqdm(data.items()):
    num_sentences += len(val['sentences'])
    for sentence in val['sentences']:
        num_words += len(sentence.split())

    for timestamp in val['timestamps']:
        start, end = timestamp
        sent_durations.append(end - start)

# plt.hist(sent_durations, range=(0,17))
# plt.savefig("./testing.png")


print("===== statistics =====")
print(f"dataset_name: {dataset_name}")
print(f"num_videos: {num_videos}")
print(f"durations: {all_duration} Ave: {all_duration/num_videos}")
print(f"max_durations: {max(durations)}")
print(f"min_durations: {min(durations)}")
print(f"num_sentences: {num_sentences} Ave: {num_sentences/num_videos}")
print(f"num_words: {num_words} Ave: {num_words/num_videos}")
print(f"max_sent_duration: {max(sent_durations)}")
print(f"min_sent_duration: {min(sent_durations)}")