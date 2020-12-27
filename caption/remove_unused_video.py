import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


def classify(video_dir_path, video_name, frame_dir_path, classify_model, threshold=0.001):

    # load model
    model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    checkpoint = torch.load(classify_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    input_size = 224

    inputs = load_video_frame(video_dir_path, video_name, frame_dir_path, input_size)
    inputs = inputs.view(1, 3, input_size, input_size)

    # classify
    inputs = inputs.to(device)
    outputs = model(inputs)
    preds = torch.sigmoid(outputs.view(-1)) > threshold

    return preds


def load_video_frame(video_dir_path, video_name, frame_dir_path, input_size):
    video_path = os.path.join(video_dir_path, video_name)
    if not os.path.exists(frame_dir_path):
        os.makedirs(frame_dir_path)

    capture = cv2.VideoCapture(video_path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture_frame = frame_count // 2

    count = 0
    retaining = True
    frame_path = ""
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count == capture_frame:
            if frame_height != input_size or frame_width != input_size:
                frame = cv2.resize(frame, (input_size, input_size))
            frame_path = os.path.join(frame_dir_path, f"{video_name}_image.jpg")
            cv2.imwrite(filename=frame_path, img=frame)
        count += 1

    capture.release()

    mean = torch.tensor([0.2223, 0.2441, 0.2484])
    std = torch.tensor([0.2064, 0.2000, 0.2207])
    normalizer = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    with open(frame_path, 'rb') as f:
        img = Image.open(f)
        frame = img.convert('RGB')
    frame = transform(frame)
    frame = normalizer(frame)

    return frame
