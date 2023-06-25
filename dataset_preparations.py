import pandas as pd
import torchvision

from torch.utils.data import DataLoader
from classes import Imagem, ImgDataset
from definitions import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_img_paths(root_path):
    stanford_training_set = {}
    for path, subdirs, files in os.walk(root_path):
        for img_name in files:
            stanford_training_set[img_name] = os.path.join(path, img_name)
    return stanford_training_set


def get_labels(root_path, csv_name):
    colnames = ['fname', 'x0', 'y0', 'x1', 'y1', 'label']
    labels = pd.read_csv(os.path.join(root_path, csv_name), names=colnames, header=None)

    return labels


def get_dataset(img_path_set, labels):
    images = []

    for index, row in labels.iterrows():
        bounding_box = (row['x0'], row['y0'], row['x1'], row['y1'])
        label = row['label']
        path = img_path_set[row['fname']]
        img = Imagem(path=path, bounding_box=bounding_box, label=label)

        images.append(img)

    return ImgDataset(images)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_data_loader(dataset, is_training_dataset, num_workers=0):
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_training_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model