import argparse
import os
import time

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoDataset
from network.C3D import C3D


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of video-clustering')

    parser.add_argument('--video-path', type=str, help='path to dataset(videos)')
    parser.add_argument('--frame-path', type=str, help='path to dataset(frames)')
    parser.add_argument('--label-path', type=str, help='path to label')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--clip-len', type=int, help='the number of frames')
    parser.add_argument('--epochs', default=200, type=int, help='epoch size(default: 200)')
    parser.add_argument('--log', type=str, help='path to log')
    parser.add_argument('--interval', type=int, help='interval of saving model')
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer_dir = os.path.join(args.log, "tensorboard")
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    writer_train = SummaryWriter(log_dir=os.path.join(writer_dir, 'train'))
    writer_valid = SummaryWriter(log_dir=os.path.join(writer_dir, 'valid'))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    transform = transforms.Compose([transforms.RandomResizedCrop(112),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    train_video_path = os.path.join(args.video_path, "train")
    valid_video_path = os.path.join(args.video_path, "valid")
    train_frame_path = os.path.join(args.frame_path, "train")
    valid_frame_path = os.path.join(args.frame_path, "valid")
    train_pre_label_path = os.path.join(args.label_path, "train.csv")
    valid_pre_label_path = os.path.join(args.label_path, "valid.csv")
    train_label_path = os.path.join(args.label_path, "new_train.csv")
    valid_label_path = os.path.join(args.label_path, "new_valid.csv")
    train_calc_path = os.path.join(args.log, 'calc/train')
    valid_calc_path = os.path.join(args.log, 'calc/valid')

    train_dataset = VideoDataset(train_video_path, train_frame_path, train_pre_label_path, train_label_path, train_calc_path, clip_len=args.clip_len, transform=transform)
    val_dataset = VideoDataset(valid_video_path, valid_frame_path, valid_pre_label_path, valid_label_path, valid_calc_path, clip_len=args.clip_len, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch,
                                            num_workers=args.workers,
                                            pin_memory=True
                                            )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch,
                                            num_workers=args.workers,
                                            pin_memory=True
                                            )
   
    print(f"{torch.cuda.device_count()} GPUs are being used.")
    model = C3D()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),  lr=0.001)

    start_epoch = 0
    model_save_dir = os.path.join(args.log, 'models')
    if os.path.exists(model_save_dir) and len(os.listdir(model_save_dir)) > 0:
        print("load pretrained model...")
        latest_path = os.path.join(model_save_dir, sorted(os.listdir(model_save_dir))[-1])
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    since = time.time()
    for epoch in range(start_epoch, args.epochs):
        print("-" * 20)
        print("Epoch {} / {}".format(epoch, args.epochs - 1))
        
        model.train()
        train_losses = 0.0
        train_corrects = 0.0
        for frames, labels in train_dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(frames)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            probs = nn.Softmax(dim=1)(pred)
            _, preds = torch.max(probs, 1)
            train_losses += loss.item() * frames.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        train_epoch_loss = torch.div(train_losses, len(train_dataloader.dataset))
        train_epoch_correct = torch.div(train_corrects.double(), len(train_dataloader.dataset))
        writer_train.add_scalar("loss", train_epoch_loss, epoch)
        writer_train.add_scalar("accuracy", train_epoch_correct, epoch)
        print("train Loss: {:.4f} Acc: {:.4f}".format(train_epoch_loss, train_epoch_correct))

        model.eval()
        valid_losses = 0.0
        valid_corrects = 0.0
        for frames, labels in val_dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                pred = model(frames)
                val_loss = criterion(pred, labels)

            probs = nn.Softmax(dim=1)(pred)
            _, preds = torch.max(probs, 1)
            valid_losses += val_loss.item() * frames.size(0)
            valid_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_losses / len(val_dataloader.dataset)
        valid_epoch_correct = valid_corrects.double() / len(val_dataloader.dataset)
        writer_valid.add_scalar("loss", valid_epoch_loss, epoch)
        writer_valid.add_scalar("accuracy", valid_epoch_correct, epoch)
        print("valid Loss: {:.4f} Acc: {:.4f}".format(valid_epoch_loss, valid_epoch_correct))

        if epoch % args.interval == 0:
            print("Saving model...")
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        },os.path.join(model_save_dir, f"Epoch-{epoch}.pkl"))
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    writer_train.close()
    writer_valid.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)