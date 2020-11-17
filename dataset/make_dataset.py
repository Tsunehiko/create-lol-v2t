import argparse
import os
import sys
import re
import pickle
import datetime
import multiprocessing
import json
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from caption.make_annotation_parallel import make_annotation_wrapper
from mmaction2.tools.data.build_rawframes_custom import extract_frame
from mmaction2.tools.data.lol.tsn_feature_extraction_custom import feature_extraction_wrapper
from logging import Logger, getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL


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
    split_ratio = {'train': 70, 'valid': 5, 'test': 25}
    # split_dir = {split: os.path.join(args.dataset, args.exp_name, split) for split in split_dict.keys()}
    dataset_dir = os.path.join(args.dataset_dir, args.exp_name)
    annotation_dir = os.path.join(dataset_dir, "annotation")
    tmp_dir = os.path.join(args.tmp_dir, args.exp_name)
    dirs = [dataset_dir, annotation_dir, tmp_dir]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    video_names = sorted([re.search(r"(.*)\.mp4", file_name).group(1) for file_name in os.listdir(args.video_dir)])
    video_num = len(video_names)
    split_nums = {}
    split_nums['train'] = int((video_num - 2) * split_ratio['train'] / 100)
    split_nums['test'] = max(int((video_num - 2) * split_ratio['test'] / 100), 1)
    split_nums['valid'] = max(video_num - split_nums['train'] - split_nums['test'], 1)
    # split_nums['train'], split_nums['test'], split_nums['valid'] = 2, 0, 0
    logger.info(f"All: {video_num}, train: {split_nums['train']}, valid: {split_nums['valid']}, test: {split_nums['test']}")

    threads_num = min(multiprocessing.cpu_count(), args.threads)
    logger.info(f'threads: {threads_num}')

    video_index = 0
    for split in split_nums.keys():
        logger.info(f"[{split}] making annotation has started.")
        split_videos = video_names[video_index: video_index + split_nums[split]]
        frame_dir = os.path.join(tmp_dir, 'frame', split)
        divided_video_dir = os.path.join(tmp_dir, 'divide', split)
        trash_dir = os.path.join(tmp_dir, 'trash', split)
        timecode_dir = os.path.join(tmp_dir, 'timecode', split)
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)
        if not os.path.exists(timecode_dir):
            os.makedirs(timecode_dir)
        results = []
        args_make_annotation_list = [(args.video_dir,
                                      args.caption_dir,
                                      divided_video_dir,
                                      frame_dir,
                                      trash_dir,
                                      timecode_dir,
                                      video,
                                      args.pyscenedetect_threshold,
                                      args.punct,
                                      args.classify_model,
                                      args.mode
                                      )
                                     for video in split_videos]

        with multiprocessing.Pool(threads_num) as pool:
            results = pool.map(make_annotation_wrapper, args_make_annotation_list)

        all_annotation_dict = {}
        for result in results:
            _, annotation_dict, _, _, _, _, _, _ = result
            all_annotation_dict.update(annotation_dict)
        with open(os.path.join(annotation_dir, split + '.json'), 'w') as f:
            json.dump(all_annotation_dict, f)

        logger.info(f"[{split}] annotation has been made.")

        logger.info(f"[{split}] making rawframes has been started.")

        for video in split_videos:
            elements_dir = os.path.join(divided_video_dir, video)
            elements = os.listdir(elements_dir)
            rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video)
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
        
        logger.info(f"[{split}] rawframes have been made.")
        logger.info(f"[{split}] rawframes_validation has been started.")

        modalities = ['RGB', 'Flow']

        for video in split_videos:
            elements_dir = os.path.join(divided_video_dir, video)
            elements = os.listdir(elements_dir)
            for element in elements:
                element_path = os.path.join(elements_dir, element)
                for modality in modalities:
                    rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video, modality, element[:-4])
                    rawframe_validation(element_path, rawframe_dir, modality)

        logger.info(f"[{split}] rawframes_validation has been done.")

        logger.info(f"[{split}] feature extraction has been started.")

        clip_lens = {'RGB': 1, 'Flow': 5}
        ckpts = {'RGB': '/home/Tanaka/generate-commentary/mmaction2/checkpoints/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20200804-9e15687e_cpu.pth',
                 'Flow': '/home/Tanaka/generate-commentary/mmaction2/checkpoints/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52_cpu.pth'}
        feature_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        args_feature_extraction_list = []
        for video in split_videos:
            elements_dir = os.path.join(divided_video_dir, video)
            elements = os.listdir(elements_dir)
            rawframe_dir = os.path.join(tmp_dir, 'rawframes', split, video)
            for element in elements:
                element_path = os.path.join(elements_dir, element)
                cap = cv2.VideoCapture(element_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for modality in modalities:
                    args_feature_extraction = (element[:-4], rawframe_dir, feature_dir, length, modality, ckpts[modality], clip_lens[modality], args.frame_interval)
                    args_feature_extraction_list.append(args_feature_extraction)
        
        with multiprocessing.Pool(threads_num) as pool:
            pool.map(feature_extraction_wrapper, args_feature_extraction_list)

        logger.info(f"[{split}] features have been extracted.")
        video_index += split_nums[split]

    logger.info("Dataset has been made.")


def rawframe_validation(video_path, frame_dir, modality):
    video_name = os.path.basename(frame_dir)
    cap = cv2.VideoCapture(video_path)
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flow = len(os.listdir(frame_dir))
    if modality == 'Flow':
        frame = frame * 2 - 2
    try:
        assert frame == flow, f'{video_name} frame:{frame} flow:{flow}'
    except AssertionError as err:
        print('AssertionError:', err)
        exit()


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
    logger = init_logger(os.path.join(args.log, args.exp_name))
    main(args, logger)
