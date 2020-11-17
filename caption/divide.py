import os
import pickle
import argparse
import cv2
import multiprocessing

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

from scenedetect.video_splitter import split_video_ffmpeg


def parse_args():
    parser = argparse.ArgumentParser(description='make caption file (.json)')

    parser.add_argument('--video-dir', type=str, help='path to video directory')
    parser.add_argument('--divided-video-dir', type=str, help='path to divided video directory (by PySceneDetect)')
    parser.add_argument('--timecode-dir', type=str, help='path to timecode pkl directory')
    parser.add_argument('--pyscenedetect-threshold', type=int, default=20, help='pyscenedetect threshold')
    parser.add_argument('--split_process')
    parser.add_argument('--threads', type=int)
    return parser.parse_args()


def main(args):

    cv2.setNumThreads(1)

    threads_num = min(multiprocessing.cpu_count(), args.threads)
    print("threads:", threads_num)
    if args.split_process:
        divide_args_list = []
        video_list = os.listdir(args.video_dir)
        video_num = len(video_list)
        split_ratio = {'train': 70, 'valid': 5, 'test': 25}
        split_nums = {}
        split_nums['train'] = int((video_num - 2) * split_ratio['train'] / 100)
        split_nums['test'] = max(int((video_num - 2) * split_ratio['test'] / 100), 1)
        split_nums['valid'] = max(video_num - split_nums['train'] - split_nums['test'], 1)
        video_index = 0
        for split in split_nums.keys():
            split_videos = video_list[video_index: video_index + split_nums[split]]
            video_elements_dir = os.path.join(args.divided_video_dir, split)
            timecode_dir = os.path.join(args.timecode_dir, split)
            if not os.path.exists(video_elements_dir):
                os.makedirs(video_elements_dir)
            if not os.path.exists(timecode_dir):
                os.makedirs(timecode_dir)
            for video in split_videos:
                video_path = os.path.join(args.video_dir, video)
                video_elements_dir_path = os.path.join(video_elements_dir, video)
                timecode_path = os.path.join(timecode_dir, video[:-4] + ".pkl")
                divide_args = (video_path, video, video_elements_dir_path, args.pyscenedetect_threshold, timecode_path)
                divide_args_list.append(divide_args)
            with multiprocessing.Pool(threads_num) as pool:
                pool.map(divide_video_wrapper, divide_args_list)
            video_index += split_nums[split]
    else:
        if not os.path.exists(args.divided_video_dir):
            os.makedirs(args.divided_video_dir)
        if not os.path.exists(args.timecode_dir):
            os.makedirs(args.timecode_dir)
        divide_args_list = []
        video_list = os.listdir(args.video_dir)
        for video in video_list:
            video_path = os.path.join(args.video_dir, video)
            video_elements_dir_path = os.path.join(args.divided_video_dir, video[:-4])
            timecode_path = os.path.join(args.timecode_dir, video[:-4] + ".pkl")
            divide_args = (video_path, video, video_elements_dir_path, args.pyscenedetect_threshold, timecode_path)
            divide_args_list.append(divide_args)
        with multiprocessing.Pool(threads_num) as pool:
            pool.map(divide_video_wrapper, divide_args_list)


def divide_video_wrapper(args):
    video_path, video, video_elements_dir_path, pyscenedetect_threshold, timecode_path = args
    print(f"[dividing]: {video} has started.")
    timecode_list = divide_video(video_path, video, video_elements_dir_path, pyscenedetect_threshold)
    print(f"[dividing]: {video} has finished.")
    save_pickle(timecode_list, timecode_path)
    print(f"[saving]: {video}'s timecode_list has been saved.")


def divide_video(video_path, video_name, save_dir_path, threshold):
    """
    Oversegment the input video using PySceneDetect

    Patameters
    ----------
    video_path : string
        path to input video
    video_name : string
        name of input video
    save_dir_path : string
        path to saving divided elements
    threshold : int
        PySceneDetect threshold
    """

    cv2.setNumThreads(1)

    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=180))
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

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    split_video_ffmpeg([video_path], scene_list, os.path.join(save_dir_path, "${VIDEO_NAME}-${SCENE_NUMBER}.mp4"), video_name[:-4], arg_override='-threads 1 -c:v libx264 -preset fast -crf 21 -c:a aac')

    timecode_list = []
    for scene in scene_list:
        start = scene[0].get_timecode()
        end = scene[1].get_timecode()
        timecode_list.append((start, end))

    return timecode_list


def save_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)