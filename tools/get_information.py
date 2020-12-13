import json
import os
import csv
import pickle

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from logging import Logger, getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL

"""
get information (duration, num of sentences, num of words, etc) about the dataset.
"""

dataset_default = "/home/Tanaka/generate-commentary/dataset"
dataset_name = "interpolation_deepsegment"
video_dir = "/home/Tanaka/generate-commentary/dataset/tmp/interpolation_deepsegment/divide"
log_dir = "./information/"
split_names = ["training", "testing", "validation"]


def main(logger):
    feature_dir = os.path.join(dataset_default, dataset_name)
    annotation_dir = os.path.join(feature_dir, "annotation/deepsegment")
    duration_dir = os.path.join(feature_dir, "duration")

    if not os.path.exists(duration_dir):
        os.makedirs(duration_dir)

    for split in split_names:

        video_dir_split = os.path.join(video_dir, split)
        logger.info("=" * 20 + f" {split} " + "=" * 20)
        videos = sorted(glob.glob(video_dir_split + '/*' * 2 + '.mp4'))
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

        num_sentences = 0
        num_words = 0

        with open(os.path.join(annotation_dir, split + ".json"), 'r') as data_file:
            data = json.load(data_file)

        sent_durations = []
        frames = []
        for vid, val in tqdm(data.items()):
            flow_feat = load_pickle(os.path.join(feature_dir, split, vid + '_flow.pkl'))
            frames.append(flow_feat.shape[0])

            num_sentences += len(val['sentences'])
            for sentence in val['sentences']:
                num_words += len(sentence.split())

            for timestamp in val['timestamps']:
                start, end = timestamp
                sent_durations.append(end - start)

        logger.info(f"dataset_name: {dataset_name}")
        logger.info(f"num_videos: {num_videos}")
        logger.info(f"durations: {all_duration} Ave: {all_duration/num_videos}")
        logger.info(f"MAX_durations: {max(durations)}")
        logger.info(f"min_durations: {min(durations)}")
        logger.info(f"num_sentences: {num_sentences} Ave: {num_sentences/num_videos}")
        logger.info(f"num_words: {num_words} Ave: {num_words/num_videos}")
        logger.info(f"MAX_sent_duration: {max(sent_durations)}")
        logger.info(f"min_sent_duration: {min(sent_durations)}")
        logger.info(f"Ave_sent_duration: {sum(sent_durations)/num_sentences}")
        logger.info(f"MAX_frame(feature): {max(frames)}")
        logger.info(f"min_frame(feature): {min(frames)}")
        logger.info(f"Ave_frame(feature): {sum(frames)/num_videos}")

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.hist(frames, range=(0, 1000), bins=20)
        ax2.hist(sent_durations, range=(0, 40), bins=40)
        plt.savefig(os.path.join(log_dir, split + ".png"))


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def init_logger(log_dir, modname=__name__):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "result.log")
    logger = getLogger('log')
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh_formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_path)
    fh.setLevel(INFO)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    return logger


if __name__ == '__main__':
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = init_logger(log_dir)
    main(logger)
