import argparse
import shutil
import json
import os
import pickle

import cv2
import webvtt
# from timecode import Timecode

from divide import divide_video
from remove_unused_video import classify
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL


def parse_args():
    parser = argparse.ArgumentParser(description='make caption file (.json)')

    parser.add_argument('--video-dir', type=str, help='path to video directory')
    parser.add_argument('--caption-dir', type=str, help='path to caption directory')
    parser.add_argument('--divided-video-dir', type=str, help='path to divided video directory (by PySceneDetect)')
    parser.add_argument('--annotation-dir', type=str, help='path to annotation directory')
    parser.add_argument('--frame-dir', type=str, help='path to frame directory in divided videos')
    parser.add_argument('--pyscenedetect-threshold', type=int, default=20, help='pyscenedetect threshold')
    parser.add_argument('--log', type=str, help='path to log')
    return parser.parse_args()


def main(args, logger):

    video_names = sorted(os.listdir(args.video_dir))
    caption_names = sorted(os.listdir(args.caption_dir))
    all_clips_num = 0
    all_duration_num = 0
    all_sentences_num = 0
    all_words_num = 0
    for i, video in enumerate(video_names):
        logger.info(f"video:{video}")
        video_path = os.path.join(args.video_dir, video)
        caption_path = os.path.join(args.caption_dir, caption_names[i])
        video_elements_dir_path = os.path.join(args.divided_video_dir, video)
        timecode_list = divide_video(video_path, video, video_elements_dir_path, args.pyscenedetect_threshold)
        
        trash_dir_path = os.path.join("./temp/trash")
        if not os.path.exists(trash_dir_path):
            os.makedirs(trash_dir_path)

        clips_num = 0
        duration_num = 0
        sentences_num = 0
        words_num = 0
        use_list = []
        unuse_list = []
        if i == 0:
            annotation_list = []
        else:
            annotation_list = load_pickle(os.path.join(args.annotation_dir, "annotation.pkl"))

        video_element_names = sorted(os.listdir(video_elements_dir_path))
        for i, video_element in enumerate(video_element_names):
            video_element_path = os.path.join(video_elements_dir_path, video_element)
            is_useful = classify(video_elements_dir_path, video_element, os.path.join(args.frame_dir, video, video_element))
            
            if is_useful:
                capture = cv2.VideoCapture(video_element_path)
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                annotation_data, sentences, words = make_caption_data(video_element, caption_path, timecode_list[i], duration)
                annotation_list.append(annotation_data)
                save_pickle(annotation_list, os.path.join(args.annotation_dir, "annotation.pkl"))
                with open(os.path.join(args.annotation_dir, "annotation.txt"), 'a') as f:
                    print(annotation_data, file=f)

                clips_num += 1
                duration_num += duration
                sentences_num += sentences
                words_num += words
                use_list.append(i)
            else:
                shutil.move(video_element_path, trash_dir_path)
                unuse_list.append(i)

        logger.info(f"clips:{clips_num}")
        logger.info(f"use  :{use_list}")
        logger.info(f"unuse:{unuse_list}")
        logger.info(f"duration:{duration_num} ave_duration:{duration_num/clips_num}")
        logger.info(f"sentences:{sentences_num} ave_sentences:{sentences_num/clips_num}")
        logger.info(f"words:{words_num} ave_words:{words_num/clips_num}")

        all_clips_num += clips_num
        all_duration_num += duration_num
        all_sentences_num += sentences_num
        all_words_num += words_num

    logger.info("Entire data")
    logger.info(f"clips:{all_clips_num} ave_clips:{all_clips_num/len(video_names)}")
    logger.info(f"duration:{all_duration_num} ave_duration:{all_duration_num/all_clips_num}")
    logger.info(f"sentences:{all_sentences_num} ave_sentences:{all_sentences_num/all_clips_num}")
    logger.info(f"words:{all_words_num} ave_words:{all_words_num/all_clips_num}")
    annotation_list = load_pickle(os.path.join(args.annotation_dir, "annotation.pkl"))
    with open(os.path.join(args.annotation_dir, 'annotation.json'), 'w') as f:
        json.dump(annotation_list, f)


def make_caption_data(video_element_name, caption_path, timecodes, duration):
    start, end = timecodes
    sentences = []
    timestamps = []
    beforeCaption = ""
    words_num = 0
    for i, caption in enumerate(webvtt.read(caption_path)):
        if i != 0 and len(caption.text.strip().splitlines()) == 1 and caption.text.strip() != beforeCaption and caption.start > start and caption.end < end:
            sentences.append(caption.text.strip())
            timestamp = [caption.start, caption.end]
            timestamps.append(timestamp)
            beforeCaption = caption.text.strip()
            words_num += len(beforeCaption.split())
    annotation = (video_element_name, {'duration':duration, 'sentences':sentences, 'timestamps':timestamps})
    return annotation, len(sentences), words_num


def save_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data
    

def init_logger(log_path, modname=__name__):
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
    args = parse_args()
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    logger = init_logger(f"{args.log}/result.log")
    main(args, logger)