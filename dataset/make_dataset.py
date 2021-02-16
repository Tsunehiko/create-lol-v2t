import argparse
import os
import sys
import re
import pickle
import datetime
import multiprocessing
import json
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from caption.make_caption import make_caption_wrapper  # noqa
from tools.validation import rawframe_validation, feature_validation  # noqa
from tools.duration import duration  # noqa
from mmaction2.tools.data.build_rawframes_custom import extract_frame  # noqa
from mmaction2.tools.data.lol.tsn_feature_extraction_custom import feature_extraction_wrapper  # noqa
from logging import Logger, getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='make caption file (.json)')

    parser.add_argument('--exp-name', type=str, help='experiment name')
    parser.add_argument('--video-dir', type=str, help='path to video directory')
    parser.add_argument('--caption-dir', type=str, help='path to caption directory')
    parser.add_argument('--tmp-dir', type=str, help='path to tmp directory')
    parser.add_argument('--dataset-dir', type=str, help='path to dataset directory (by PySceneDetect)')
    parser.add_argument('--log', type=str, help='path to log')
    parser.add_argument('--pyscenedetect-threshold', type=int, default=20, help='pyscenedetect threshold')
    parser.add_argument('--threads', type=int, help='num of threads')
    parser.add_argument('--punct', type=str, default='deepsegment', help='Sentence Boundary Detection Library. two options: deepsegment | fastpunct')
    parser.add_argument('--flow-type', type=str, default=None, choices=[None, 'tvl1', 'warp_tvl1', 'farn', 'brox'], help='flow type to be generated')
    parser.add_argument('--task', type=str, default='both', help='denseflow task', choices=['both', 'flow', 'rgb'])
    parser.add_argument('--new-short', type=int, default=0, help='resize image short side length keeping ratio')
    parser.add_argument('--new-width', type=int, default=0, help='resize image width')
    parser.add_argument('--new-height', type=int, default=0, help='resize image height')
    parser.add_argument('--use-opencv', action='store_true', help='Whether to use opencv to extract rgb frames')
    parser.add_argument('--input-frames', action='store_true', help='Whether to extract flow frames based on rgb frames')
    parser.add_argument('--frame-interval', type=int, default=16, help='feature extraction frame interval')
    parser.add_argument('--classify-model', type=str)
    parser.add_argument('--mode', type=str, choices=['wide', 'interpolation'])
    return parser.parse_args()


def main(args, logger):
    split_ratio = {'training': 75, 'validation': 5, 'testing': 20}
    # split_dir = {split: os.path.join(args.dataset, args.exp_name, split) for split in split_dict.keys()}
    dataset_dir = os.path.join(args.dataset_dir, args.exp_name)
    annotation_dir = os.path.join(dataset_dir, "annotation", args.punct)
    duration_dir = os.path.join(dataset_dir, "duration")
    tmp_dir = os.path.join(args.tmp_dir, args.exp_name)
    dirs = [dataset_dir, annotation_dir, tmp_dir]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    video_names = sorted([re.search(r"(.*)\.mp4", file_name).group(1) for file_name in os.listdir(args.video_dir)])
    # video_names = video_names[0:2]
    video_num = len(video_names)
    split_nums = {}
    split_nums['training'] = int((video_num - 2) * split_ratio['training'] / 100)
    split_nums['testing'] = max(int((video_num - 2) * split_ratio['testing'] / 100), 1)
    split_nums['validation'] = max(video_num - split_nums['training'] - split_nums['testing'], 1)
    split_nums['training'], split_nums['testing'], split_nums['validation'] = 81, 20, 8
    logger.info(f"All: {video_num}, training: {split_nums['training']}, validation: {split_nums['validation']}, testing: {split_nums['testing']}")

    threads_num = min(multiprocessing.cpu_count(), args.threads)
    logger.info(f'threads: {threads_num}')

    video_index = 0
    for split in ["training", "validation", "testing"]:
        split_videos = video_names[video_index: video_index + split_nums[split]]
        frame_dir = os.path.join(tmp_dir, 'frame', split)
        divided_video_dir = os.path.join(tmp_dir, 'divide', split)
        trash_dir = os.path.join(tmp_dir, 'trash', split)
        timecode_dir = os.path.join(tmp_dir, 'timecode', split)
        annotation_path = os.path.join(annotation_dir, split + '.json')
        tmp_annotation_dir = os.path.join(tmp_dir, 'annotation', split)
        if not os.path.exists(annotation_path):
            logger.info(f"[{split}] making annotation has started.")
            if not os.path.exists(trash_dir):
                os.makedirs(trash_dir)
            if not os.path.exists(timecode_dir):
                os.makedirs(timecode_dir)
            args_make_caption_list = [(args.video_dir,
                                       args.caption_dir,
                                       divided_video_dir,
                                       frame_dir,
                                       trash_dir,
                                       timecode_dir,
                                       video,
                                       args.pyscenedetect_threshold,
                                       args.punct,
                                       args.classify_model,
                                       args.mode,
                                       tmp_annotation_dir
                                       )
                                      for video in split_videos]

            with multiprocessing.Pool(threads_num) as pool:
                pool.map(make_caption_wrapper, args_make_caption_list)

            all_annotation_dict = {}
            tmp_annotation_paths = sorted(os.listdir(tmp_annotation_dir))
            for tmp_annotation_path in tmp_annotation_paths:
                with open(os.path.join(tmp_annotation_dir, tmp_annotation_path), 'rb') as f:
                    annotation_dict = json.load(f)
                all_annotation_dict.update(annotation_dict)
            with open(annotation_path, 'w') as f:
                json.dump(all_annotation_dict, f)

            logger.info(f"[{split}] annotation has been made.")
        else:
            logger.info(f'[{split}] annotation has already been made.')

        logger.info(f"[{split}] making rawframes has been started.")
        for video in split_videos:
            elements_dir = os.path.join(divided_video_dir, video)
            elements = os.listdir(elements_dir)
            rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video)
            if not os.path.exists(rawframe_dir):
                for element in elements:
                    element_path = os.path.join(elements_dir, element)
                    args_extract_frame = (element_path,
                                          rawframe_dir,
                                          args.flow_type,
                                          args.task,
                                          args.new_short,
                                          args.new_width,
                                          args.new_height,
                                          args.use_opencv,
                                          args.input_frames)
                    _ = extract_frame(args_extract_frame)
                logger.info(f"[{split}] [{video}] rawframes have been made.")
            else:
                logger.info(f'[{split}] [{video}] rawframes has already been made.')
        logger.info(f"[{split}] making rawframes has been finished.")
        
        logger.info(f"[{split}] rawframes_validation has started.")

        modalities = ['RGB', 'Flow']

        all_err_video = []
        for video in tqdm(split_videos):
            elements_dir = os.path.join(divided_video_dir, video)
            elements = os.listdir(elements_dir)
            for element in elements:
                element_path = os.path.join(elements_dir, element)
                for modality in modalities:
                    rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video, modality, element[:-4])
                    err_count, err_video = rawframe_validation(element_path, rawframe_dir, modality)
                    if err_count > 0:
                        all_err_video.append(err_video)
        if len(all_err_video) > 0:
            logger.error(f'rawframes validation error:{all_err_video}')
            exit()

        logger.info(f"[{split}] rawframes_validation has been done.")

        feature_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(feature_dir):
            logger.info(f"[{split}] feature extraction has started.")
            clip_lens = {'RGB': 1, 'Flow': 5}
            ckpts = {'RGB': '/home/Tanaka/generate-commentary/mmaction2/checkpoints/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20200804-9e15687e_cpu.pth',
                     'Flow': '/home/Tanaka/generate-commentary/mmaction2/checkpoints/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52_cpu.pth'}
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            args_feature_extraction_list = []
            for video in tqdm(split_videos):
                elements_dir = os.path.join(divided_video_dir, video)
                elements = os.listdir(elements_dir)
                rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video)
                for element in elements:
                    element_path = os.path.join(elements_dir, element)
                    cap = cv2.VideoCapture(element_path)
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for modality in modalities:
                        args_feature_extraction = (element[:-4],
                                                   rawframe_dir,
                                                   feature_dir,
                                                   length,
                                                   modality,
                                                   ckpts[modality],
                                                   clip_lens[modality],
                                                   args.frame_interval)
                        args_feature_extraction_list.append(args_feature_extraction)
            
            with multiprocessing.Pool(threads_num) as pool:
                pool.map(feature_extraction_wrapper, args_feature_extraction_list)

            logger.info(f"[{split}] features have been extracted.")
        else:
            logger.info(f'[{split}] features have already been extracted.')

        logger.info(f'[{split}] feature validation has started.')
        err_videos = feature_validation(divided_video_dir, feature_dir)
        if len(err_videos) > 0:
            logger.error(f'feature extraction validation error: {err_videos}')
            exit()
        logger.info(f'[{split}] feature validation has done.')

        video_index += split_nums[split]

    logger.info("Making duration has started.")
    duration(os.path.join(tmp_dir, "divide"), duration_dir)
    logger.info("duration_frame.csv has been made.")

    logger.info("Dataset has been made.")


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
    logger = init_logger(os.path.join(args.log, args.exp_name, date))
    main(args, logger)
