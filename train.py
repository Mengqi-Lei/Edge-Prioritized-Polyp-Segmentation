import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, label_dictionary, \
    mask_to_bbox
from my_model import EPSS
from metrics import DiceLoss, DiceBCELoss, MultiClassBCE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from run_engine import load_data, train, evaluate, DATASET


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    """ Parameter Setting"""
    # dataset
    # dataset_name= 'CVC-ClinicDB'
    # dataset_name= 'Kvasir-SEG'
    dataset_name = 'kvasir-sessile'

    seed_ = 42

    # seed
    my_seeding(seed_)

    # Hyperparameters
    image_size = 256
    size = (image_size, image_size)
    batch_size = 4
    num_epochs = 150
    lr = 1e-4
    early_stopping_patience = 50

    # the weight of edge loss and mi loss
    alpha = 1  # for edge
    beta = 1  # for MI

    resume_path = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"prewitt_{dataset_name}_lr{lr}_a{alpha}_b{beta}_{current_time}"

    # Directories
    base_dir = "../data"  # base directory
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join("ablation", dataset_name, folder_name)  # path to save log and model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    # 创建并打开训练日志文件
    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    # 记录日期和时间
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    # 记录超参数
    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Alpha: {alpha}\nBeta: {beta}\n"
    hyperparameters_str += f"Seed: {seed_}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = EPSS()

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    # 输出模型的参数量(M)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device, alpha, beta)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device, alpha, beta)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
