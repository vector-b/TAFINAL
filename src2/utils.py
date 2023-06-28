import json
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm.auto import tqdm

from definitions import OUT_DIR, WIDTH, HEIGHT


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
    prog_bar = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            images, targets = data

            images = list(image.to(DEVICE) for image in images)
            result_prediction = model(images)

            for i in range(len(result_prediction)):
                json_obj = {"boxes": [], "labels": [], "scores": [], "image_id": targets[i]["image_id"].item()}
                for box, label, score in zip(result_prediction[i]['boxes'],
                                             result_prediction[i]['labels'],
                                             result_prediction[i]['scores']):
                    json_obj["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
                    json_obj["labels"].append(label.item())
                    json_obj["scores"].append(score.item())

                predictions.append(json_obj)

    time_id = time.time()
    with open(os.path.join(OUT_DIR, f"predictions_{time_id}.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    return predictions


def video_writer(imgs_dir, imgs_info, predictions):
    class_names = {1: "Banhista", 2: "Salva-vidas", 3: "Barco", 4: "Banhista no barco", 5: "Salva-vidas no barco",
                   6: "Colete"}
    colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 51), 4: (0, 255, 255), 5: (255, 102, 255),
              6: (51, 153, 255)}
    shape = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24
    current_video = None
    out = None

    print("Video creation begin!")
    for current_prediction in predictions:
        info = imgs_info[current_prediction["image_id"]]
        if current_video != info['source']['video']:
            if out is not None:
                out.release()
                print(f"Video '{current_video}' saved.")

            current_video = info['source']['video']
            out = cv2.VideoWriter(f"./results/{current_video}", fourcc, fps, shape)

        image = cv2.imread(f"{imgs_dir}/{info['file_name'].replace('.png', '.jpg')}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, shape)

        counter = 0
        for j in range(len(current_prediction["boxes"])):
            score = current_prediction["scores"][j]
            box = current_prediction["boxes"][j]
            category = current_prediction["labels"][j]

            if score < 0.75 or category == 7:
                continue
            else:
                counter += 1

            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            cv2.rectangle(image_resized, start_point, end_point, color=colors[category], thickness=2)

            cv2.putText(
                image_resized, f"{class_names[category]}: {int(score*100)}%", (int(box[0]), int(box[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=colors[category], thickness=2)

        cv2.putText(image_resized, str(counter), (20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                    color=(0, 0, 0), thickness=4)

        out.write(image_resized)

    out.release()
    print(f"Video '{current_video}' saved.")



