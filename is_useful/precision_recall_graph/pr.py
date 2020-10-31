import argparse
import time
import os
import copy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import FrameDataset

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of video-clustering')

    parser.add_argument('--video-path', type=str, help='path to dataset(videos)')
    parser.add_argument('--frame-path', type=str, help='path to dataset(frames)')
    parser.add_argument('--label-path', type=str, help='path to label')
    parser.add_argument('--model-path', type=str, help='path to model')
    parser.add_argument('--log', type=str, help='path to log')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--model-name', type=str, help='model name')
    return parser.parse_args()


def main(args, logger):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model, input_size = initialize_model(args.model_name, 1)
    logger.info(f"{torch.cuda.device_count()} GPUs are being used.")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    cudnn.benchmark = True
    logger.debug("Load pretrained model...")
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    video_path = os.path.join(args.video_path, 'valid')
    frame_path = os.path.join(args.frame_path, 'valid')
    label_path = os.path.join(args.label_path, "valid.csv")
    calc_path = os.path.join(args.log, "calc/valid")

    dataset = FrameDataset(video_path, frame_path, label_path, calc_path, input_size, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, shuffle=False)

    model.eval()

    labels = torch.zeros(len(dataloader.dataset))
    outputs = torch.zeros(len(dataloader.dataset))
    for i, (inputs, label, vnames) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
        labels[i] = label.data
        outputs[i] = output.data
    
    thresholds = np.arange(0.0,1.05,0.01)
    precisions = np.zeros(len(thresholds))
    recalls= np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        preds = torch.sigmoid(outputs) >= threshold

        tp = torch.sum((preds.data == 1.0) & (labels.data == 1.0))
        tn = torch.sum((preds.data == 0.0) & (labels.data == 0.0))
        fp = torch.sum((preds.data == 1.0) & (labels.data == 0.0))
        fn = torch.sum((preds.data == 0.0) & (labels.data == 1.0))

        assert tp + fp + tn + fn == len(dataset)

        precision = tp.double() / (tp.double() + fp.double())
        precision = precision.to('cpu').detach().numpy().copy()
        recall = tp.double() / (tp.double() + fn.double())
        recall = recall.to('cpu').detach().numpy().copy()
        if np.isnan(precision):
            precision = 1.0
        precisions[i] = precision
        recalls[i] = recall

    print("-"*20)
    print(precisions)
    print("-"*20)
    print(recalls)

    plt.plot(recalls, precisions, marker='o')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    plt.grid(True)
    plt.savefig(os.path.join(args.log, "pr.png"))


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet18':
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'resnet101':
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'resnext':
        """ ResNext50_32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'alexnet':
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vgg':
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'squeezenet':
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == 'inception':
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        logger.warning("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


def init_logger(log_path, modname=__name__):
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
    logger = init_logger(f"{args.log}/result.log")
    main(args, logger)