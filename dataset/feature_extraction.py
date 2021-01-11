import argparse
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mmaction2.tools.data.lol.tsn_feature_extraction_custom import feature_extraction  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--video_name', default='')
    parser.add_argument('--video_path')
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--modality')
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument('--clip-len', type=int, default=1, help='clip length')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=16,
        help='the sampling frequency of frame in the untrimed video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feature_extraction(args.video_name, args.input_dir, args.output_dir, length, args.modality, args.ckpt, args.clip_len, args.frame_interval)


if __name__ == '__main__':
    main()
