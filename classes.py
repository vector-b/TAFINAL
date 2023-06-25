import cv2
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class Imagem():
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
        


''' # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0'''

class ImgDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.width = 512
        self.height = 512
        pass
        
    def __getitem__(self,idx):
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
        
        box = (x0_final, y0_final, x1_final, y1_final)
        
        return image_resized, box
