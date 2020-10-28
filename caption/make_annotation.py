import argparse
from os import times
import shutil
import json
import csv
import os
import re
import pickle
import datetime

import cv2
import webvtt
from timecode import Timecode
from fastpunct import FastPunct
from nltk.tokenize import sent_tokenize

from divide import divide_video
from remove_unused_video import classify
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL


def parse_args():
    parser = argparse.ArgumentParser(description='make caption file (.json)')

    parser.add_argument('--video-dir', type=str, help='path to video directory')
    parser.add_argument('--caption-dir', type=str, help='path to caption directory' )
    parser.add_argument('--divided-video-dir', type=str, help='path to divided video directory (by PySceneDetect)')
    parser.add_argument('--annotation-dir', type=str, help='path to annotation directory')
    parser.add_argument('--frame-dir', type=str, help='path to frame directory in divided videos')
    parser.add_argument('--timecode-dir', type=str, help='path to timecode pkl directory')
    parser.add_argument('--pyscenedetect-threshold', type=int, default=20, help='pyscenedetect threshold')
    parser.add_argument('--log', type=str, help='path to log')
    return parser.parse_args()


def main(args, logger, date):

    video_names = sorted([re.search(r"(.*)\.mp4", file_name).group(1) for file_name in os.listdir(args.video_dir)])
    all_clips_num = 0
    all_duration_num = 0
    all_sentences_num = 0
    all_words_num = 0
    for i, video in enumerate(video_names):
        logger.info(f"video:{video}")
        video_name = video + ".mp4"
        video_path = os.path.join(args.video_dir, video_name)
        caption_path = os.path.join(args.caption_dir, (video + ".en.vtt"))
        annotation_dir = os.path.join(args.annotation_dir, date)
        if not os.path.exists(annotation_dir):
            os.makedirs(annotation_dir)
        video_elements_dir_path = os.path.join(args.divided_video_dir, date, video_name)
        timecode_list = divide_video(video_path, video_name, video_elements_dir_path, args.pyscenedetect_threshold)
        timecode_dir = os.path.join(args.timecode_dir, date)
        if not os.path.exists(timecode_dir):
            os.makedirs(timecode_dir)
        save_pickle(timecode_list, os.path.join(timecode_dir, video + ".pkl"))
        
        trash_dir_path = os.path.join("./tmp/test/trash/series")
        if not os.path.exists(trash_dir_path):
            os.makedirs(trash_dir_path)

        clips_num = 0
        duration_num = 0
        sentences_num = 0
        words_num = 0
        use_list = []
        unuse_list = []
        if i == 0:
            annotation_dict = {}
        else:
            annotation_dict = load_pickle(os.path.join(args.annotation_dir, date, "annotation.pkl"))

        video_element_names = sorted(os.listdir(video_elements_dir_path))
        fastpunct = FastPunct('en')
        for i, video_element in enumerate(video_element_names):
            video_element_path = os.path.join(video_elements_dir_path, video_element)
            is_useful = classify(video_elements_dir_path, video_element, os.path.join(os.path.join(args.frame_dir, "series"), video_name, video_element))
            
            if is_useful:
                capture = cv2.VideoCapture(video_element_path)
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                annotation_data, sentences, words = make_caption_data(video_element[:-4], caption_path, timecode_list[i], duration, fps, fastpunct)
                annotation_dict.update(annotation_data)
                save_pickle(annotation_dict, os.path.join(args.annotation_dir, date, "annotation.pkl"))
                with open(os.path.join(args.annotation_dir, date, "annotation.txt"), 'a') as f:
                    print(annotation_data, file=f)
                with open(os.path.join(args.annotation_dir, date, "duration_frame.csv"), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_element, duration, frame_count])
                
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
    all_annotation_dict = load_pickle(os.path.join(args.annotation_dir, date, "annotation.pkl"))
    with open(os.path.join(args.annotation_dir, date, 'annotation.json'), 'w') as f:
        json.dump(all_annotation_dict, f)


def make_caption_data(video_element_name, caption_path, timecodes, duration, fps, fastpunct):
    # 動画の最初と最後
    start, end = Timecode(fps, timecodes[0]), Timecode(fps, timecodes[1])
    start_sec = timecode_to_sec(start)

    captions = webvtt.read(caption_path)
    captions[1].start = captions[0].start
    captions = captions[1:]

    start_cap = timecode_to_sec(Timecode(fps, captions[0].start))
    words_num = 0
    caption_dict_list = []
    joined_sentence = ""
    for i, caption in enumerate(captions):
        caption_dict = {}
        if caption.start > start and caption.end < end and i % 2 == 0:
            start_cap = timecode_to_sec(Timecode(fps, caption.start))
            end_cap = timecode_to_sec(Timecode(fps, captions[i + 1].end))
            sentence = caption.text.strip().splitlines()[0]
            joined_sentence += sentence + ' '
            caption_dict['start'] = start_cap - start_sec
            caption_dict['end'] = end_cap - start_sec
            caption_dict['sentence'] = sentence
            caption_dict_list.append(caption_dict)
            words_num += len(sentence.split())
    
    sentences = []
    divided_sentences = []
    joined_sentence_sequence = len(joined_sentence)
    while joined_sentence_sequence > 0:
        punct_sentences = fastpunct.punct([joined_sentence[:390]], batch_size=32)
        divided_sentences = sent_tokenize(punct_sentences[0])
        sentences.extend(divided_sentences[:-1])
        joined_sentence = divided_sentences[-1] + joined_sentence[390:]
        joined_sentence_sequence = joined_sentence_sequence - 390 + len(divided_sentences[-1])
    sentences.append(divided_sentences[-1])
    
    tmp_start = caption_dict_list[0]['start']
    timestamps = []
    sentence_i = 0
    num_sentences = len(sentences)
    for i, caption_dict in enumerate(caption_dict_list):
        if sentence_i == num_sentences:
            break
        sentence = sentences[sentence_i].translate(str.maketrans({',': None, '.': None, "'": None, ' ': None})).lower()
        caption = caption_dict['sentence'].translate(str.maketrans({',': None, '.': None, "'": None, ' ': None, '-': None})).lower()
        if caption not in sentence:
            timestamps.append([tmp_start, caption_dict['end']])
            tmp_start = caption_dict['start']
            sentence_i += 1
        elif sentence[:-len(caption)] == caption:
            timestamps.append([tmp_start, caption_dict['end']])
            tmp_start = caption_dict_list[i + 1]['start']
            sentence_i += 1
                
    annotation = {video_element_name: {'duration': duration, 'timestamps': timestamps, 'sentences': sentences}}
    return annotation, len(sentences), words_num


def timecode_to_sec(timecode):
    return timecode.hrs * 3600 + timecode.mins * 60 + timecode.secs + timecode.frs


def save_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


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
    args = parse_args()
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    date = str(datetime.date.today())
    logger = init_logger(os.path.join(args.log, date))
    main(args, logger, date)