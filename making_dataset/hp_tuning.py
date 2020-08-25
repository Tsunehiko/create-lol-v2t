import time
import os

from tqdm import tqdm
import numpy as np

import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from dataset import FrameDataset

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):

    name = 'optuna'
    video_path = "./data/"
    frame_path = "./2D/frame"
    label_path = "./label"
    log_path = os.path.join("./2D/results/", name)
    workers = 12
    model_name = "resnext"
    batch_size = 64
    seed = 31
    epochs = 10
    feature_extract = False

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = init_logger(f"{log_path}/result.log")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model, input_size = initialize_model(model_name, 1, feature_extract)
    logger.info(f"{torch.cuda.device_count()} GPUs are being used.")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    cudnn.benchmark = True

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    threshold = trial.suggest_uniform("threshold", 0.1, 0.9)

    transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    video_paths = {x: os.path.join(video_path, x) for x in ['train', 'valid']}
    frame_paths = {x: os.path.join(frame_path, x) for x in ['train', 'valid']}
    label_paths = {x: os.path.join(label_path, f"{x}.csv") for x in ['train', 'valid']}
    calc_paths = {x: os.path.join(log_path, f"calc/{x}") for x in ['train', 'valid']}

    datasets = {x: FrameDataset(video_paths[x], frame_paths[x], label_paths[x], calc_paths[x], input_size, transform=transform) for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True) for x in ['train', 'valid']}

    criterion = nn.BCEWithLogitsLoss()
    
    precision = train(trial, model, dataloaders, criterion, optimizer, epochs, logger, is_inception=(model_name=="inception"), threshold=threshold)

    return precision


def train(trial, model, dataloaders, criterion, optimizer, num_epochs, logger, is_inception=False, threshold=0.5):
    since = time.time()
    precision = 0

    for epoch in range(num_epochs):
        # logger.info("-" * 20)
        # logger.info("Epoch {} / {}".format(epoch+1, num_epochs))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for inputs, labels, vnames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        labels = labels.type_as(outputs)
                        loss = criterion(outputs.view(-1), labels)
                    
                    preds = torch.sigmoid(outputs.view(-1)) > threshold

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)
                tp += torch.sum((preds.data == 1.0) & (labels.data == 1.0))
                tn += torch.sum((preds.data == 0.0) & (labels.data == 1.0))
                fp += torch.sum((preds.data == 1.0) & (labels.data == 0.0))
                fn += torch.sum((preds.data == 0.0) & (labels.data == 0.0))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            precision = tp.double() / (tp.double() + fp.double())
            recall = tp.double() / (tp.double() + fn.double())
            f1 = 2 * tp.double() / (2 * tp.double() + fp + fn)

            # logger.info("{} Loss: {:.7f} Acc:{:.7f} Precision:{:.7f} Recall:{:.7f} F1:{:.7f}"
            #         .format(phase, epoch_loss, epoch_acc, precision, recall, f1))

        trial.report(precision, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return precision


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
    logger.setLevel(WARNING)

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

def main():
    study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100, timeout=600)
    study.optimize(objective, n_trials=50)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    name = 'optuna'
    main()