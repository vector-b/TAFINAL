import os

BATCH_SIZE = 6 # increase / decrease according to GPU memeory
WIDTH = 640
HEIGHT = 360
NUM_CLASSES = 8
NUM_WORKERS = 10

OUT_DIR = './results'

BASE_DATASET_IMAGES_PATH = "../dataset"
STANFORD_SET_PATH = os.path.join(BASE_DATASET_IMAGES_PATH, "stanford_by_class")
STANFORD_TRAIN_PATH = os.path.join(STANFORD_SET_PATH, "car_data/car_data/train/")
STANFORD_VALIDATION_PATH = os.path.join(STANFORD_SET_PATH, "car_data/car_data/test/")

MODEL_SAVE_PATH = os.path.join(OUT_DIR, "model.pt")

SEA_DRONES_MOT_SET_PATH = os.path.join(BASE_DATASET_IMAGES_PATH, "sea_drones_see_tracking")
SEA_DRONES_TRAIN_PATH = os.path.join(SEA_DRONES_MOT_SET_PATH, 'Compressed/train')
SEA_DRONES_VALIDATION_PATH = os.path.join(SEA_DRONES_MOT_SET_PATH, 'Compressed/val')
SEA_DRONES_VIDEOS_PATH = os.path.join(SEA_DRONES_MOT_SET_PATH, 'Compressed/test')