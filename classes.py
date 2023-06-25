from typing import List
import albumentations
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np


class Imagem(object):
    def __init__(self, path, bounding_box, label) -> None:
        self.path  = path
        self.bounding_box = bounding_box
        #self.cv2_image = cv2.imread(path)
        self.label = label
        pass
    
    def show_img(self):
        start_point = self.bounding_box[:2]
        end_point = self.bounding_box[-2:]
        cv2.rectangle(self.cv2_image, start_point, end_point, color=(0,255,0), thickness=2)
        cv2.imshow('Car', self.cv2_image)


def get_training_transform():
    return albumentations.Compose([
        albumentations.Flip(0.5),
        albumentations.RandomRotate90(0.5),
        albumentations.MotionBlur(p=0.2),
        albumentations.MedianBlur(blur_limit=3, p=0.1),
        albumentations.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


class ImgDataset(Dataset):
    def __init__(self, img_list: List[Imagem]):
        self.img_list: List[Imagem] = img_list
        self.width = 512
        self.height = 512
        self.transforms = get_training_transform()
        pass
        
    def __getitem__(self, idx):
        img_obj = self.img_list[idx]
        image = cv2.imread(img_obj.path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        image_width = image.shape[1]
        image_height = image.shape[0]

        bbox = img_obj.bounding_box

        x0_final = (bbox[0]/image_width) * self.width
        x1_final = (bbox[1]/image_width) * self.width
        y0_final = (bbox[2]/image_height) * self.height
        y1_final = (bbox[3]/image_height) * self.height
        
        boxes = torch.as_tensor([[x0_final, y0_final, x1_final, y1_final]], dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor([img_obj.label], dtype=torch.int64)
        # box = [x0_final, y0_final, x1_final, y1_final]
        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=labels)
        image_resized = sample['image']
        target['boxes'] = torch.Tensor(sample['boxes'])

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