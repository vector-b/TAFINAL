import cv2

class Dataset():
    def __init__(self, path, bounding_box) -> None:
        self.path  = path
        self.bounding_box = bounding_box
        self.cv2_image = cv2.imread(path)
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