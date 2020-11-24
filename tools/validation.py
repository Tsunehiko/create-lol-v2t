import pickle
import os
import glob
import cv2


def rawframe_validation(video_path, frame_dir, modality):
    video_name = os.path.basename(frame_dir)
    cap = cv2.VideoCapture(video_path)
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flow = len(os.listdir(frame_dir))
    if modality == 'Flow':
        frame = frame * 2 - 2
    err_count = 0
    err_video = ""
    try:
        assert frame == flow, f'{video_name} {modality} frame:{frame} flow:{flow}'
    except AssertionError as err:
        print('AssertionError:', err)
        err_count = 1
        err_video = video_name
    
    return err_count, err_video


def feature_validation(video_dir, feature_dir):
    videos = sorted(glob.glob(video_dir + '/*' * 2 + '.mp4'))
    err_videos = []
    for video in videos:
        video_name = os.path.basename(video[:-4])
        flow_feat = load_pickle(os.path.join(feature_dir, video_name + '_flow.pkl'))
        rgb_feat = load_pickle(os.path.join(feature_dir, video_name + '_rgb.pkl'))
        try:
            assert flow_feat.shape[0] == rgb_feat.shape[0], f'[{video_name}] Flow: {flow_feat.shape[0]}, RGB: {rgb_feat.shape[0]}'
        except AssertionError as err:
            print('AssertionError:', err)
            err_videos.append(video_name)
    return err_videos


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data
