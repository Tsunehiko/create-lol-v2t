import time
import os
import pickle

from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from logging import getLogger

logger = getLogger('log').getChild(__name__)

class VideoDataset(data.Dataset):

    def __init__(self, video_dir, frame_dir, pre_label_path, label_path, calc_path, clip_len=16, preprocess=False, transform=None):
        self.video_dir = video_dir
        self.frame_dir = frame_dir
        self.pre_label_path = pre_label_path
        self.label_path = label_path
        self.clip_len = clip_len
        self.transform = transform
        
        self.resize_height = 112
        self.resize_width = 112
        self.crop_size = 112

        self.vnames = []
        self.fnames = []
        self.fpaths = []

        labels = pd.read_csv(self.pre_label_path)
        count = 0
        for vname in sorted(os.listdir(self.video_dir)):
            video_path = os.path.join(self.video_dir, vname)
            self.vnames.append(vname)
            video_elements = []

            drop = 0
            for v_el in sorted(os.listdir(video_path)):
                v_el_path = os.path.join(video_path, v_el)
                capture = cv2.VideoCapture(v_el_path)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count < 16:
                    labels.drop([count])
                    drop += 1
                else:
                    frame_dir = os.path.join(self.frame_dir, vname, v_el)
                    self.fpaths.append(frame_dir)
                    video_elements.append(frame_dir)

                count += 1

            print(f"{drop} elements in {vname} were dropped.")
            self.fnames.append(video_elements)
        
        labels.to_csv(self.label_path, index=False)

        print(f"Num of videos: {len(self.vnames)}")
        for i, v_el in enumerate(self.fnames):
            print(f"Num of {self.vnames[i]}'s elements: {len(v_el)}")

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of the dataset, this will take long, but it will be done only once.')
            self.preprocess()

        mean_path = os.path.join(calc_path, 'mean.pt')
        if os.path.exists(mean_path):
            print("Load mean...")
            self.mean = torch.load(mean_path)
        else:
            if not os.path.exists(calc_path):
                os.makedirs(calc_path)
            self.mean = self.calc_mean(mean_path)

        std_path = os.path.join(calc_path, 'std.pt')
        if os.path.exists(std_path):
            print("Load std...")
            self.std = torch.load(std_path)
        else:
            if not os.path.exists(calc_path):
                os.makedirs(calc_path)
            self.std = self.calc_std(std_path)

        label_csv = pd.read_csv(self.label_path)
        print(label_csv['label'].value_counts())

        assert len(label_csv) == len(self.fpaths)

    def __len__(self):
        return sum(len(f) for f in self.fnames)

    def __getitem__(self, index):
        tensor = torch.ones(())
        buf_tr = tensor.new_empty((self.clip_len, 3, self.resize_height, self.resize_width))
        buf = self.load_frames(self.fpaths[index])
        if self.transform is not None:
            for i, frame in enumerate(buf):
                frame_tr = self.transform(frame)
                buf_tr[i] = self.normalize(frame_tr)
        buf_tr = buf_tr.permute(1,0,2,3)

        label = pd.read_csv(self.label_path).iat[index, 3]

        return buf_tr, label

    def check_preprocess(self):
        if not os.path.exists(self.frame_dir):
            return False
        
        return True

    def preprocess(self):
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

        for vname in self.vnames:
            print(f"Preprocessing `{vname}`")
            video_path = os.path.join(self.video_dir, vname)
            frame_dir = os.path.join(self.frame_dir, vname)
            for v_el in tqdm(os.listdir(video_path)):
                video_el_path = os.path.join(video_path, v_el)
                frame_el_dir = os.path.join(frame_dir, v_el)
                self.process_video(video_el_path, frame_el_dir)

    def process_video(self, video_path, frame_dir):

        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        capture = cv2.VideoCapture(video_path)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        extract_frequency = frame_count // 16

        count = 0
        i = 0
        retaining = True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            # if count % extract_frequency == 0:
            #     if frame_height != self.resize_height or frame_width != self.resize_width:
            #         frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            #     cv2.imwrite(filename=os.path.join(base_path, f"image{i}.jpg"), img=frame)
            #     i += 1

            if count < self.clip_len:
                if frame_height != self.resize_height or frame_width != self.resize_width:
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(frame_dir, f"image{i}.jpg"), img=frame)
                i += 1

            count += 1

        capture.release()

    def calc_mean(self, save_path):
        print("Calculating mean...")
        mean = torch.zeros((3, self.resize_height, self.resize_width))
        for j, path in enumerate(tqdm(self.fpaths)):
            buf = self.load_frames(path)
            if self.transform is not None:
                for i, frame in enumerate(buf):
                    mean += self.transform(frame)
        mean = mean / len(self.fpaths)
        torch.save(mean, save_path)
        return mean

    def calc_std(self, save_path):
        print("Calculting std...")
        std = torch.zeros((3, self.resize_height, self.resize_width))
        for j, path in enumerate(tqdm(self.fpaths)):
            buf = self.load_frames(path)
            if self.transform is not None:
                for i, frame in enumerate(buf):
                    std += torch.pow((self.transform(frame) - self.mean), 2)
        std = torch.sqrt(std / len(self.fpaths))
        torch.save(std, save_path)
        return std

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = []
        for i, frame_name in enumerate(frames):
            if i >= self.clip_len:
                continue
            frame = self.pil_loader(frame_name)
            buffer.append(frame)

        return buffer

    def pil_loader(self, path):
        """Loads an image.
        Args:
            path (string): path to image file
        Returns:
            Image
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def normalize(self, buf):
        buf = (buf - self.mean) / (self.std + 10e-5)
        return buf

    def save_pickle(self, obj, file):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            pickle.load(f)


class FrameDataset(data.Dataset):

    def __init__(self, video_dir, frame_dir, label_path, calc_path, input_size, preprocess=False, transform=None):
        self.video_dir = video_dir
        self.frame_dir = frame_dir
        self.label_path = label_path
        self.transform = transform
        
        self.resize_height = input_size
        self.resize_width = input_size
        self.crop_size = input_size

        self.vnames = []
        self.vnames_per_el = []
        self.fnames = []
        self.fpaths = []
        for vname in sorted(os.listdir(self.video_dir)):
            video_path = os.path.join(self.video_dir, vname)
            self.vnames.append(vname)
            video_elements = []
            for v_el in sorted(os.listdir(video_path)):
                frame_dir = os.path.join(self.frame_dir, vname, v_el)
                self.fpaths.append(frame_dir)
                video_elements.append(frame_dir)
                self.vnames_per_el.append(v_el)
            self.fnames.append(video_elements)

        logger.info(f"Num of videos: {len(self.vnames)}")
        for i, v_el in enumerate(self.fnames):
            logger.info(f"Num of {self.vnames[i]}'s elements: {len(v_el)}")

        if (not self.check_preprocess()) or preprocess:
            logger.info('Preprocessing of the dataset, this will take long, but it will be done only once.')
            self.preprocess()


        mean_path = os.path.join(calc_path, 'mean.pt')
        if os.path.exists(mean_path):
            logger.debug("Load mean...")
            self.mean = torch.load(mean_path)
        else:
            if not os.path.exists(calc_path):
                os.makedirs(calc_path)
            self.mean = self.calc_mean(mean_path)

        std_path = os.path.join(calc_path, 'std.pt')
        if os.path.exists(std_path):
            logger.debug("Load std...")
            self.std = torch.load(std_path)
        else:
            if not os.path.exists(calc_path):
                os.makedirs(calc_path)
            self.std = self.calc_std(std_path)
        
        self.normalizer = transforms.Normalize(mean=self.mean, std=self.std)

        label_csv = pd.read_csv(self.label_path)
        logger.info(label_csv['label'].value_counts())

        assert len(label_csv) == len(self.fpaths)

    def __len__(self):
        return sum(len(f) for f in self.fnames)

    def __getitem__(self, index):
        frame = self.load_frame(self.fpaths[index])
        if self.transform is not None:
            frame = self.transform(frame)
        # frame = self.normalize(frame)
        frame = self.normalizer(frame)

        label = pd.read_csv(self.label_path).iat[index, 3]

        return frame, label, self.vnames_per_el[index]

    def check_preprocess(self):
        if not os.path.exists(self.frame_dir):
            return False
        
        return True

    def preprocess(self):
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

        for vname in self.vnames:
            logger.debug(f"Preprocessing `{vname}`")
            video_path = os.path.join(self.video_dir, vname)
            frame_dir = os.path.join(self.frame_dir, vname)
            for v_el in tqdm(os.listdir(video_path)):
                video_el_path = os.path.join(video_path, v_el)
                frame_el_dir = os.path.join(frame_dir, v_el)
                self.process_video(video_el_path, frame_el_dir)

    def process_video(self, video_path, frame_dir):

        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        capture = cv2.VideoCapture(video_path)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        capture_frame = frame_count // 2

        count = 0
        retaining = True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count == capture_frame:
                if frame_height != self.resize_height or frame_width != self.resize_width:
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(frame_dir, f"image.jpg"), img=frame)
            
            count += 1

        capture.release()

    def calc_mean(self, save_path):
        logger.debug("Calculating mean...")
        mean = torch.zeros(3)
        for j, path in enumerate(tqdm(self.fpaths)):
            frame = self.load_frame(path)
            if self.transform is not None:
                frame = self.transform(frame)
                mean += frame.sum(dim=[1,2]) / (frame.size(1) * frame.size(2))
        mean = mean / len(self.fpaths)
        logger.info(f"mean: {mean}")
        torch.save(mean, save_path)
        return mean

    def calc_std(self, save_path):
        logger.debug("Calculting std...")
        std = torch.zeros(3)
        for j, path in enumerate(tqdm(self.fpaths)):
            frame = self.load_frame(path)
            if self.transform is not None:
                frame = self.transform(frame)
                for i in range(3):
                    std[i] += torch.pow((frame[i] - self.mean[i]), 2).sum() / (frame.size(1) * frame.size(2))
        std = torch.sqrt(std / len(self.fpaths))
        logger.info(f"std: {std}")
        torch.save(std, save_path)
        return std

    def load_frame(self, file_dir):
        frame_path = os.path.join(file_dir, os.listdir(file_dir)[0])
        frame = self.pil_loader(frame_path)

        return frame

    def pil_loader(self, path):
        """Loads an image.
        Args:
            path (string): path to image file
        Returns:
            Image
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def normalize(self, buf):
        buf = (buf - self.mean) / (self.std + 10e-5)
        return buf

    def save_pickle(self, obj, file):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            pickle.load(f)