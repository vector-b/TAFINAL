from typing import List
from torchvision.transforms import Compose, ToTensor, Resize
import cv2
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from definitions import WIDTH, HEIGHT


class Imagem(object):
    def __init__(self, path, bounding_box, label, areas=None) -> None:
        self.path  = path
        self.bounding_box = bounding_box
        #self.cv2_image = cv2.imread(path)
        self.label = label
        self.areas = areas
        pass
    
    def show_img(self):
        start_point = self.bounding_box[:2]
        end_point = self.bounding_box[-2:]
        cv2.rectangle(self.cv2_image, start_point, end_point, color=(0,255,0), thickness=2)
        cv2.imshow('Car', self.cv2_image)


def get_training_transform():
    return Compose([
        ToTensor(),
    ])


class ImgDataset(Dataset):
    def __init__(self, img_list: List[Imagem]):
        self.img_list: List[Imagem] = img_list
        self.width = WIDTH
        self.height = HEIGHT
        self.transforms = Compose([
            Resize((self.height, self.width)),
            ToTensor(),
        ])
        pass
        
    def __getitem__(self, idx):
        img_obj = self.img_list[idx]
        image = Image.open(img_obj.path)

        image_resized = self.transforms(image)

        image_width = image.width
        image_height = image.height

        boxes = []
        for bbox in img_obj.bounding_box:
            x_min = bbox[0] / (image_width / self.width)
            y_min = bbox[1] / (image_height / self.height)
            x_max = x_min + bbox[2] / (image_width / self.width)
            y_max = y_min + bbox[3] / (image_height / self.height)

            boxes.append([x_min, y_min, x_max, y_max])
        if len(boxes) == 0:
            print(img_obj.path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(img_obj.areas, dtype=torch.float32)

        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(img_obj.label, dtype=torch.int64)
        
        # box = [x0_final, y0_final, x1_final, y1_final]
        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd,
                  'org_w': torch.as_tensor(image_width, dtype=torch.int64),
                  'org_h': torch.as_tensor(image_height, dtype=torch.int64)}

        return image_resized, target


    def __len__(self):
        return len(self.img_list)


class Averager(object):
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0