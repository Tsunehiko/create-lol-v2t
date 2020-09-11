import os
import csv

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

from scenedetect.video_splitter import split_video_ffmpeg


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

    split_video_ffmpeg([video_path], scene_list, os.path.join(save_dir_path,"${VIDEO_NAME}-${SCENE_NUMBER}.mp4"), video_name[:-4])

    timecode_list = []
    for scene in scene_list:
        start = scene[0].get_timecode()
        end = scene[1].get_timecode()
        timecode_list.append((start, end))

    return timecode_list