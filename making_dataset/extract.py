from __future__ import print_function
import os
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import FrameDataset

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager
# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of video-clustering')

    parser.add_argument('--video-path', type=str, help='path to dataset(videos)')
    parser.add_argument('--frame-path', type=str, help='path to dataset(frames)')
    parser.add_argument('--label-path', type=str, help='path to label')
    parser.add_argument('--log', type=str, help='path to log')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--epochs', default=200, type=int, help='epoch size(default: 200)')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--feature-extract', help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params')
    parser.add_argument('--interval', type=int, help='interval of saving model')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--lr', type=float, help='optimizer learning rate')
    return parser.parse_args()


def main():
    video_path = "./videos/valid"
    save_dir = "./new_data/valid"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists("./stats"):
        os.makedirs("./stats")

    for i, vpath in enumerate(sorted(os.listdir(video_path))):
        fpath = os.path.join(video_path, vpath)
        make_dataset(fpath, vpath, save_dir)
        for j in range(length):
            if start <= j and end >= j:
                writer.writerow([i, vpath, j, 1])
            else:
                writer.writerow([i, vpath, j, 0])


def make_elements(video_path, video_name, save_dir):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    base_timecode = video_manager.get_base_timecode()

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = 'stats/%s.stats.csv' % video_name

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.

    finally:
        video_manager.release()

    video_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    split_video_ffmpeg([video_path], scene_list, os.path.join(video_dir,"${VIDEO_NAME}-${SCENE_NUMBER}.mp4"), video_name)


if __name__ == '__main__':
    main()