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

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def convert_to_edge(mask, method):
    """
Extracts the edges of the image according to the specified method
    """

    if method == "sobel":
        grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
        return binary_output.astype(np.uint8)

    elif method == "laplacian":
        laplacian = cv2.Laplacian(mask, cv2.CV_64F)
        _, binary_output = cv2.threshold(np.abs(laplacian), 50, 255, cv2.THRESH_BINARY)
        return binary_output.astype(np.uint8)

    elif method == "canny":
        edges = cv2.Canny(mask, 100, 200)
        return edges

    elif method == "scharr":
        grad_x = cv2.Scharr(mask, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(mask, cv2.CV_64F, 0, 1)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
        return binary_output.astype(np.uint8)

    elif method == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = cv2.filter2D(mask, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(mask, cv2.CV_64F, kernel_y)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
        return binary_output.astype(np.uint8)

    else:
        raise ValueError("Unknown method for edge detection")


def load_names(path, file_path):
    """
    :param path:
    :param file_path:
    :return:
    """
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
    return images, masks


def load_names_fused_train(path, file_path):
    """
    :param path:
    :param file_path:
    :return:
    """
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "train", "images", name) + ".png" for name in data]
    masks = [os.path.join(path, "train", "masks", name) + ".png" for name in data]
    return images, masks


def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)  # 读取图像和掩码的路径

    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, convert_edge="prewitt"):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.convert_edge = convert_edge
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        # edge 直接深拷贝mask
        edge = mask.copy()
        edge = convert_to_edge(edge, self.convert_edge)

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, edge=edge)
            image = augmentations["image"]
            mask = augmentations["mask"]
            edge = augmentations["edge"]

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Edge """
        edge = cv2.resize(edge, self.size)
        edge = np.expand_dims(edge, axis=0)
        edge = edge / 255.0

        return image, (mask, edge)

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device, alpha=0.5, beta=0.5):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x), (y1, y2)) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred, edge_pred, loss_mi1, loss_mi2, loss_mi3, loss_mi4 = model(x)

        loss_mask = loss_fn(mask_pred, y1)
        loss_edge = loss_fn(edge_pred, y2)
        loss_mi = (loss_mi1 + loss_mi2 + loss_mi3 + loss_mi4) / 4.0

        loss = loss_mask + alpha * loss_edge + beta * loss_mi
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device, alpha=0.5, beta=0.5):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, edge_pred, loss_mi1, loss_mi2, loss_mi3, loss_mi4 = model(x)

            loss1 = loss_fn(mask_pred, y1)
            loss2 = loss_fn(edge_pred, y2)
            loss3 = (loss_mi1 + loss_mi2 + loss_mi3 + loss_mi4) / 4.0
            loss = loss1 + alpha * loss2 + beta * loss3
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
