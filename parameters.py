import sys, os 
sys.path.append('..')
sys.path.append(os.getcwd())

import utils.utils as CU
import utils.dataloaders as CD


# Cityscapes
CITYSCAPES_TRAINING_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/train'
CITYSCAPES_TRAINING_LABELS = CITYSCAPES_TRAINING_IMAGES.replace('images', 'labels')
CITYSCAPES_VALIDATION_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/val'
CITYSCAPES_VALIDATION_LABELS = CITYSCAPES_VALIDATION_IMAGES.replace('images', 'labels')
CITYSCAPES_TEST_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/test'
CITYSCAPES_TEST_LABELS = CITYSCAPES_TEST_IMAGES.replace('images', 'labels')

# BDD100K locations 
BDD100K_DATA_DIR = '/mnt/ml-data-storage/jens/BDD100K/images'
BDD100K_LABEL_DIR = BDD100K_DATA_DIR.replace('images', 'labels')

# KITTI-360


# A2D2


# Classes in both Cityscapes, BDD100K, KITTI-360 and (partly) in A2D2
CLASSES = ['road', 'sidewalk', 'building', 'wall', 
       'fence', 'pole', 'traffic light', 'traffic sign', 
       'vegetation', 'terrain', 'sky', 
       'person', 'rider', 
       'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


# Extra
DEVICE = 'cuda'
CLASS_DISTRIBUTION_FOLDER = './class_distribution/'