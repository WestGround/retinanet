# import the necessary packages
import os

# Set the dataset base path here
BASE_PATH = "/content/retinanet/dataset"
TRAIN_BASE_PATH = "/content/retinanet/dataset/train"
TEST_BASE_PATH = "/content/retinanet/dataset/test"

# build the path to the annotations and input images
TRAIN_ANNOT_PATH = os.path.sep.join([TRAIN_BASE_PATH, 'annotations'])
TRAIN_IMAGES_PATH = os.path.sep.join([TRAIN_BASE_PATH, 'images'])
TEST_ANNOT_PATH = os.path.sep.join([TEST_BASE_PATH, 'annotations'])
TEST_IMAGES_PATH = os.path.sep.join([TEST_BASE_PATH, 'images'])

#  build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, 'train.csv'])
TEST_CSV = os.path.sep.join([BASE_PATH, 'test.csv'])

# build the path to the output classes CSV files
CLASSES_CSV = os.path.sep.join([BASE_PATH, 'classes.csv'])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, 'predictions'])
