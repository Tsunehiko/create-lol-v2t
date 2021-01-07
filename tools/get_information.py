import json
import os
import pickle
import datetime

import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from collections import Counter
from logging import Logger, getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL

"""
get information (duration, num of sentences, num of words, etc) about the dataset.
"""

dataset_default = "/home/Tanaka/generate-commentary/dataset"
dataset_name = "lol"
# dataset_default = "/home/Tanaka/densecap/data/anet/"
# dataset_name = "ActivityNet"
feature_dir = os.path.join(dataset_default, dataset_name)
if dataset_name == "ActivityNet":
    split_names = ["training", "validation"]
    annotation_dir = os.path.join(feature_dir, "annotation")
    video_dir_list = []
else:
    split_names = ["training", "testing", "validation"]
    annotation_dir = os.path.join(feature_dir, "annotation/deepsegment")
    video_dir_list = [
        "/home/Tanaka/generate-commentary/dataset/tmp/short/divide",
        "/home/Tanaka/generate-commentary/dataset/tmp/large/divide"]
log_dir = os.path.join("./information/", dataset_name, str(datetime.date.today()))


def main(logger):

    all_split_num_videos = 0
    all_split_durations = []
    all_split_num_sentences = 0
    all_split_num_words = 0
    all_split_sent_durations = []
    all_split_frames = []
    all_split_num_words_list = []

    for split in split_names:

        logger.info("=" * 20 + f" {split} " + "=" * 20)
        videos = []
        for video_dir in video_dir_list:
            video_dir_split = os.path.join(video_dir, split)
            videos += sorted(glob.glob(video_dir_split + '/*' * 2 + '.mp4'))
        num_videos = 0
        all_duration = 0
        durations = []
        fps_list = []

        num_videos = len(videos)
        video_names_feature = []
        if dataset_name != 'ActivityNet':
            for video in tqdm(videos):
                video_name = os.path.basename(video)
                video_names_feature.append(video_name[:-4])
                capture = cv2.VideoCapture(video)
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps_list.append(fps)
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
        video_names_annotation = []
        features = []
        for vid, val in tqdm(data.items()):
            video_names_annotation.append(vid)
            if dataset_name == 'ActivityNet':
                flow_feat_path = os.path.join(
                    feature_dir, split, vid + '_bn.npy')
            else:
                flow_feat_path = os.path.join(feature_dir,
                                              split,
                                              vid + '_flow.pkl')
            if os.path.exists(flow_feat_path):
                if dataset_name == 'ActivityNet':
                    flow_feat = torch.from_numpy(
                        np.load(flow_feat_path)).float()
                else:
                    flow_feat = load_pickle(flow_feat_path)
                features.append(flow_feat)
                frames.append(flow_feat.shape[0])
            else:
                logger.warning(f'{vid} feature is missing.')

            num_sentences += len(val['sentences'])
            for sentence in val['sentences']:
                num_words += len(sentence.split())
                all_split_num_words_list.append(len(sentence.split()))

            for timestamp in val['timestamps']:
                start, end = timestamp
                sent_durations.append(end - start)

        if dataset_name == 'ActivityNet':
            dur_file = "/home/Tanaka/densecap/data/anet/anet_duration_frame.csv"
            with open(dur_file) as f:
                for line in f:
                    vid_name, vid_dur, _ = [li.strip()
                                            for li in line.split(',')]
                    if vid_name in video_names_annotation:
                        durations.append(float(vid_dur))
                        all_duration += float(vid_dur)
        else:
            not_annotation_videos = set(video_names_feature) - set(video_names_annotation)
            logger.info(f'not annotation videos: {not_annotation_videos}')

        num_annotations = len(video_names_annotation)  # これが実質的な動画の数

        all_split_num_videos += num_annotations
        all_split_durations += durations
        all_split_num_sentences += num_sentences
        all_split_num_words += num_words
        all_split_sent_durations += sent_durations
        all_split_frames += frames

        logger.info(f"dataset_name: {dataset_name}")
        logger.info(f"num_videos: {num_videos}")
        logger.info(f'num_annotation: {num_annotations}')
        logger.info(f'num_feature: {len(features)}')
        logger.info(f"fps: {Counter(fps_list).most_common()}")
        logger.info(
            f"durations: {all_duration} Ave: {all_duration/num_annotations} ({min(durations)} ~ {max(durations)})")
        logger.info(
            f"num_sentences: {num_sentences} Ave: {num_sentences/num_annotations}")
        logger.info(f"num_words: {num_words} Ave: {num_words/num_annotations}")
        logger.info(
            f"Ave_sent_duration: {sum(sent_durations)/num_sentences} ({min(sent_durations)} ~ {max(sent_durations)})")
        logger.info(
            f"Ave_frame(feature): {sum(frames)/num_annotations} ({min(frames)} ~ {max(frames)})")

        fig = plt.figure()
        fig.suptitle(f'{split}')
        ax1 = fig.add_subplot(131, xlabel='frames', title='video_frames')
        ax2 = fig.add_subplot(
            132,
            xlabel='duration(sec)',
            title='video_durations')
        ax3 = fig.add_subplot(
            133,
            xlabel='duration(sec)',
            title='sentence_duration')
        ax1.hist(frames, range=(0, 1000), bins=20)
        ax2.hist(durations, range=(0, 250), bins=25)
        ax3.hist(sent_durations, range=(0, 20), bins=20)
        plt.savefig(os.path.join(log_dir, split + ".svg"))

    logger.info("=" * 20 + " All Splits " + "=" * 20)
    logger.info(f"durations: {sum(all_split_durations)} (sec) {sum(all_split_durations)/60} (min), Ave: {sum(all_split_durations)/all_split_num_videos} ({min(all_split_durations)} ~ {max(all_split_durations)})")
    logger.info(f"num_sentences: {all_split_num_sentences}, Ave: {all_split_num_sentences/all_split_num_videos}")
    logger.info(f"num_words: {all_split_num_words}, Ave: {all_split_num_words/all_split_num_sentences}")
    logger.info(f"Ave_sent_duration: {sum(all_split_sent_durations)/all_split_num_sentences} (sec) ({min(all_split_sent_durations)} ~ {max(all_split_sent_durations)})")
    logger.info(f"Ave_frame: {sum(all_split_frames)/all_split_num_videos} ({min(all_split_frames)} ~ {max(all_split_frames)})")

    fig = plt.figure()
    ax1 = fig.add_subplot(111, xlabel='frames')
    ax1.hist(all_split_frames, range=(0, max(all_split_frames)), bins=20)
    plt.savefig(os.path.join(log_dir, "frames" + ".pdf"))
    fig = plt.figure()
    ax2 = fig.add_subplot(
        111,
        xlabel='duration(second)')
    ax2.hist(all_split_durations, range=(0, max(all_split_durations)), bins=25)
    plt.savefig(os.path.join(log_dir, "video_duration" + ".pdf"))
    fig = plt.figure()
    ax3 = fig.add_subplot(
        111,
        xlabel='duration(second)')
    ax3.hist(all_split_sent_durations, range=(0, 20), bins=20)
    plt.savefig(os.path.join(log_dir, "sentence_durations" + ".pdf"))
    fig = plt.figure()
    ax4 = fig.add_subplot(
        111,
        xlabel='Words per sentence')
    ax4.hist(all_split_num_words_list, range=(0, max(all_split_num_words_list)), bins=20)
    plt.savefig(os.path.join(log_dir, "sentence_words" + ".pdf"))


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
    sh_formatter = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_path)
    fh.setLevel(INFO)
    fh_formatter = Formatter(
        '%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = init_logger(log_dir)
    main(logger)
