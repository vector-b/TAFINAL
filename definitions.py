import os

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_CLASSES = 197
NUM_WORKERS = 10

BASE_DATASET_IMAGES_PATH = "./dataset"
STANFORD_SET_PATH = os.path.join(BASE_DATASET_IMAGES_PATH, "stanford_by_class")
STANFORD_TRAIN_PATH = os.path.join(STANFORD_SET_PATH, "car_data/car_data/train/")
STANFORD_VALIDATION_PATH = os.path.join(STANFORD_SET_PATH, "car_data/car_data/test/")
MODEL_SAVE_PATH = "./model.pt"