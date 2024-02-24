import os, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from my_model import EPSS
from utils import create_dir, seeding
from utils import calculate_metrics
from run_engine import load_data


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def process_edge(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)

    y_pred = y_pred > 0.001
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def print_score(metrics_score):
    jaccard = metrics_score[0] / len(test_x)  #
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    f2 = metrics_score[5] / len(test_x)

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")


def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()

            mask_pred, edge_pred, loss_mi1, loss_mi2, loss_mi3, loss_mi4 = model(image)

            p1 = mask_pred
            p2 = edge_pred

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p1 = process_edge(p1)
            p2 = process_edge(p2)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, p1, line, p2], axis=1)

        cv2.imwrite(f"{save_path}/all/{name}.jpg", cat_images)

        cv2.imwrite(f"{save_path}/mask/{name}.jpg", p1)
        cv2.imwrite(f"{save_path}/edge/{name}.jpg", p2)

    print_score(metrics_score_1)
    # 保存到文件
    # 创建文件
    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / len(test_x):1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / len(test_x):1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / len(test_x):1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / len(test_x):1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / len(test_x):1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / len(test_x):1.4f}\n")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    # Set the data set name
    # dataset_name = 'CVC-ClinicDB'
    dataset_name = 'Kvasir-SEG'
    # dataset_name = 'kvasir-sessile'

    seeding(42)
    size = (256, 256)
    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EPSS(256, 256)
    model = model.to(device)
    checkpoint_path = "Path to your checkpoint.pth file"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "../data/{}/".format(dataset_name)
    (train_x, train_y), (test_x, test_y) = load_data(path)

    # Set the save path
    save_path = f"results/{dataset_name}/MyModel"

    create_dir(f"{save_path}/all")
    create_dir(f"{save_path}/edge")
    create_dir(f"{save_path}/mask")
    evaluate(model, save_path, test_x, test_y, size)
