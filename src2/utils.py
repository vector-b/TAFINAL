import json
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from definitions import OUT_DIR, WIDTH, HEIGHT
from torch.cuda.memory import change_current_allocator


def save_loss_plot(train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')


def create_predictions(model, data_loader):
    DEVICE = torch.device('cuda')
    model.eval()

    predictions = []
    previewed_images = []
    print("pred")
    with torch.no_grad():
        for images, targets in data_loader:
            previewed_images.extend(images)
            images = list(image.to(DEVICE) for image in images)
            result_prediction = model(images)

            json_obj = {"boxes": [], "labels": [], "scores": [], "image_id": targets[0]["image_id"].item()}
            for box, label, score in zip(result_prediction[0]['boxes'],
                                         result_prediction[0]['labels'],
                                         result_prediction[0]['scores']):
                json_obj["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
                json_obj["labels"].append(label.item())
                json_obj["scores"].append(score.item())

            print(result_prediction)
            predictions.append(json_obj)

    time_id = time.time()
    with open(os.path.join(OUT_DIR, f"predictions_{time_id}.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    return predictions, previewed_images


def video_writer(imgs_dir, imgs_info, predictions):
    '''
    one color for each category (excluding category nr 7)
        1: swimmer
        2: floater
        3: boat
        4: swimmer on boat
        5: floater on boat
        6: life jacket
        7: ignored
    '''
    colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 51), 4: (0, 255, 255), 5: (255, 102, 255),
              6: (51, 153, 255)}
    shape = WIDTH, HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 2
    out = cv2.VideoWriter("./results/output.avi", fourcc, fps, shape)
    for current_prediction in predictions:
        info = imgs_info[current_prediction["image_id"]]
        print(f"{imgs_dir}/{info['file_name']}")
        image = cv2.imread(f"{imgs_dir}/{info['file_name'].replace('.png', '.jpg')}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (WIDTH, HEIGHT))

        for j in range(len(current_prediction["boxes"])):
            if current_prediction["scores"][j] < 0.5:
                print("Too low percentage, skipping.")
                continue
            box = current_prediction["boxes"][j]
            category = current_prediction["labels"][j]

            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            cv2.rectangle(image_resized, start_point, end_point, color=colors[category], thickness=2)
        cv2.imwrite("./results/img.png", image_resized)
        out.write(image_resized)
    out.release()



