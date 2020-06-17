from __future__ import print_function
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

def main():
    video_path = "./test_data/valid"
    save_dir = "./test_data/valid/video"
    csv_path = "./test_data/label/valid.csv"
    timecodes = [
        ["00:09:32.000", "00:09:34.000", "00:45:09.000", "00:45:11.000"], 
        ["00:13:07.000", "00:13:09.000", "00:41:32.000", "00:41:34.000"], 
        ["00:09:41.000", "00:09:43.000", "00:43:12.000", "00:43:14.000"],
        ["00:11:15.000", "00:11:17.000", "00:40:13.000", "00:40:15.000"],
        ["00:07:55.000", "00:07:57.000", "00:32:07.000", "00:32:09.000"],
        ["00:16:38.000", "00:16:40.000", "00:41:42.000", "00:41:44.000"],
        ["00:23:39.000", "00:23:41.000", "00:55:15.000", "00:55:17.000"],
        ["00:08:18.000", "00:08:20.000", "00:32:59.000", "00:33:01.000"],
        ["00:08:32.000", "00:08:34.000", "00:41:59.000", "00:42:01.000"],
        ["00:08:52.000", "00:08:54.000", "00:45:09.000", "00:45:11.000"],
        ["00:17:07.000", "00:17:09.000", "00:50:09.000", "00:50:11.000"],
        ["00:07:27.000", "00:07:29.000", "00:47:00.000", "00:47:02.000"],
        ["00:20:18.000", "00:20:20.000", "00:46:26.000", "00:46:28.000"],
        ["00:15:07.000", "00:15:09.000", "00:48:03.000", "00:48:05.000"],
        ["00:08:55.000", "00:08:57.000", "00:48:42.000", "00:48:44.000"],
        ["00:13:17.000", "00:13.19.000", "00:54:56.000", "00:54:58.000"],
        ["00:08:56.000", "00:08:58.000", "00:42:36.000", "00:42:38.000"],
        ["00:09:50.000", "00:09:52.000", "00:53:14.000", "00:53:16.000"],
        ["00:22:15.000", "00:22:17.000", "00:52:48.000", "00:52:50.000"],
        ["00:10:47.000", "00:10:49.000", "00:56:43.000", "00:56:45.000"],
        ["00:07:25.000", "00:07:27.000", "00:30:02.000", "00:30:04.000"],
        ["00:13:41.000", "00:13:43.000", "00:46:20.000", "00:46:22.000"],
        ["00:13:06.000", "00:13:08.000", "00:42:49.000", "00:42:51.000"],
        ["00:07:49.000", "00:07:51.000", "00:42:23.000", "00:42:25.000"],
        ["00:06:49.000", "00:06:51.000", "00:43:02.000", "00:43:04.000"],
        ["00:22:55.000", "00:22:57.000", "00:54:23.000", "00:54:25.000"],
        ["00:14:30.000", "00:14:32.000", "00:48:34.000", "00:48:36.000"],
        ["00:09:20.000", "00:09:22.000", "00:46:33.000", "00:46:35.000"],
        ["00:13:28.000", "00:13:30.000", "00:37:48.000", "00:37:50.000"],
        ["00:11:44.000", "00:11:46.000", "00:31:51.000", "00:31:53.000"],
        ["00:11:24.000", "00:11:26.000", "00:42:46.000", "00:42:48.000"],
        ["00:22:21.000", "00:22:23.000", "00:48:04.000", "00:48:05.000"],
        ["00:11:03.000", "00:11:05.000", "00:35:30.000", "00:35:32.000"],
        ["00:08:37.000", "00:08:39.000", "00:37:24.000", "00:37:26.000"],
        ["00:08:59.000", "00:09:01.000", "00:46:40.000", "00:46:42.000"],
        ["00:10:55.000", "00:10:57.000", "00:37:55.000", "00:37:57.000"],
        ["00:11:47.000", "00:11:49.000", "00:39:13.000", "00:39:15.000"],
        ["00:18:01.000", "00:18:03.000", "00:43:47.000", "00:43:49.000"],
        ["00:10:00.000", "00:10:02.000", "00:45:13.000", "00:45:15.000"],
        ["00:11:28.000", "00:11:30.000", "01:03:45.000", "01:03:47.000"],
        ["00:13:46.000", "00:13:48.000", "00:41:26.000", "00:41:28.000"],
        ["00:10:55.000", "00:10:57.000", "00:37:48.000", "00:37:50.000"],
        ["00:12:28.000", "00:12:30.000", "00:46:25.000", "00:46:27.000"],
        ["00:21:15.000", "00:21:17.000", "01:02:36.000", "01:02:38.000"],
        ["00:16:51.000", "00:16:53.000", "01:01:40.000", "01:01:42.000"],
        ["00:13:24.000", "00:13:26.000", "00:38:14.000", "00:38:16.000"],
        ["00:18:54.000", "00:18:56.000", "00:54:35.000", "00:54:36.000"],
        ["00:19:10.000", "00:19:12.000", "00:47:06.000", "00:47:08.000"],
        ["00:14:25.000", "00:14:27.000", "00:55:52.000", "00:55:54.000"],
        ["00:39:49.000", "00:39:51.000", "01:11:57.000", "01:11:58.000"]
        ]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists("./stats"):
        os.makedirs("./stats")

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'video_num', 'label'])
        for i, vpath in enumerate(sorted(os.listdir(video_path))):
            print(vpath)
            fpath = os.path.join(video_path, vpath)
            start, end, length = make_dataset(fpath, vpath, timecodes[i], save_dir)
            for j in range(length):
                if start <= j and end >= j:
                    writer.writerow([vpath, j, 1])
                else:
                    writer.writerow([vpath, j, 0])


def make_dataset(video_path, video_name, timecodes, save_dir):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())
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

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

        start_timecode = ""
        start_content_val = 0
        end_timecode = ""
        end_content_val = 0
        metric_keys = sorted(list(stats_manager._registered_metrics.union(stats_manager._loaded_metrics)))
        frame_keys = sorted(stats_manager._frame_metrics.keys())
        for frame_key in frame_keys:
            frame_timecode = base_timecode + frame_key
            timecode = frame_timecode.get_timecode()
            if timecode > timecodes[0] and timecode < timecodes[1]:
                content_val = stats_manager.get_metrics(frame_key, metric_keys)[0]
                if start_content_val < content_val:
                    start_content_val = content_val
                    start_timecode = timecode
            if timecode > timecodes[2] and timecode < timecodes[3]:
                content_val = stats_manager.get_metrics(frame_key, metric_keys)[0]
                if end_content_val < content_val:
                    end_content_val = content_val
                    end_timecode = timecode
        threshold = min(start_content_val, end_content_val)

    finally:
        video_manager.release()

    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    base_timecode = video_manager.get_base_timecode()

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

        start_video_num = 0
        end_video_num = 0
        for i, scene in enumerate(scene_list):
            if scene[0].get_timecode() >= start_timecode and start_video_num == 0:
                start_video_num = i + 1
                print(f"start video: {start_video_num}")
            if scene[1].get_timecode() >= end_timecode and end_video_num == 0:
                end_video_num = i - 1
                print(f"end video: {end_video_num}")

    finally:
        video_manager.release()

    video_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    split_video_ffmpeg([video_path], scene_list, os.path.join(video_dir,"${VIDEO_NAME}-${SCENE_NUMBER}.mp4"), video_name)

    return start_video_num, end_video_num, len(scene_list)

if __name__ == '__main__':
    main()