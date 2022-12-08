import cv2 
import numpy as np
from torch.utils.data import Dataset as Dataset

import sys
sys.path.append('..')
import time

from tqdm.notebook import tqdm
import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize




class A2D2(Dataset):
    """
    Audi Autonomous Driving Dataset (A2D2). Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available from class_list.json included in the dataset. 
    
    Args:
        images (list): List of all paths to the images for the dataset 
        labels (list): List of all paths to the labels for the dataset 
        class_values (list): A selection of classes to extract from segmentation mask if not all is to be used. 
            (e.g. ['car'] or ['car', 'person'], etc.) 
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    ORIGINAL_CLASSES = [
        "unlabeled",
        "car",
        "bicycle",
        "pedestrian",
        "truck",
        "small vehicle",
        "traffic signal",
        "traffic sign",
        "utility vehicle",
        "sidebars",
        "speed bumper",
        "curbstone",
        "solid line",
        "irrelevant signs",
        "road blocks",
        "tractor",
        "non-drivable street",
        "zebra crossing",
        "obstacles",
        "poles",
        "restricted area",
        "animals",
        "grid structure",
        "signal corpus",
        "drivable cobblestone",
        "electronic traffic",
        "slow driving area",
        "nature object",
        "parking area",
        "sidewalk",
        "ego vehicle",
        "painted driving instruction",
        "traffic guide",
        "dashed line",
        "rd normal street",
        "sky",
        "building",
        "blurred area",
        "rain dirt"
    ]
    
    CLASSES = [
        "unlabeled", 
        "car", 
        "bicycle", 
        "person", 
        "truck", 
        "motorcycle", 
        "traffic light", 
        "traffic sign", 
        "utility vehicle", 
        "sidebars", 
        "speed bumper", 
        "curbstone", 
        "road", 
        "irrelevant signs", 
        "road blocks", 
        "tractor", 
        "non-drivable street", 
        "zebra crossing", 
        "obstacle", 
        "pole", 
        "restricted area", 
        "animals", 
        "fence", 
        "traffic light", 
        "road", 
        "electronic traffic", 
        "road", 
        "vegetation", 
        "parking", 
        "sidewalk", 
        "ego vehicle", 
        "road", 
        "traffic guide", 
        "road", 
        "road", 
        "sky", 
        "building", 
        "blurred area", 
        "rain dirt", 
        "wall",  # Included, as exists in BDD100K
        "terrain",  # Included, as exists in BDD100k
        "bus",
        "train",
        "rider", 
   ]
    
    INSTANCE_LABELS = {
        (255, 0, 0): 1,
        (200, 0, 0): 1,
        (150, 0, 0): 1,
        (128, 0, 0): 1,
        (182, 89, 6): 2,
        (150, 50, 4): 2,
        (90, 30, 1): 2,
        (90, 30, 30): 2,
        (204, 153, 255): 3,
        (189, 73, 155): 3,
        (239, 89, 191): 3,
        (255, 128, 0): 4,
        (200, 128, 0): 4,
        (150, 128, 0): 4,
        (0, 255, 0): 5,
        (0, 200, 0): 5,
        (0, 150, 0): 5,
        (0, 128, 255): 6,
        (30, 28, 158): 6,
        (60, 28, 100): 6,
        (0, 255, 255): 7,
        (30, 220, 220): 7,
        (60, 157, 199): 7,
        (255, 255, 0): 8,
        (255, 255, 200): 8,
        (233, 100, 0): 9,
        (110, 110, 0): 10,
        (128, 128, 0): 11,
        (255, 193, 37): 12,
        (64, 0, 64): 13,
        (185, 122, 87): 14,
        (0, 0, 100): 15,
        (139, 99, 108): 16,
        (210, 50, 115): 17,
        (255, 0, 128): 18,
        (255, 246, 143): 19,
        (150, 0, 150): 20,
        (204, 255, 153): 21,
        (238, 162, 173): 22,
        (33, 44, 177): 23,
        (180, 50, 180): 24,
        (255, 70, 185): 25,
        (238, 233, 191): 26,
        (147, 253, 194): 27,
        (150, 150, 200): 28,
        (180, 150, 200): 29,
        (72, 209, 204): 30,
        (200, 125, 210): 31,
        (159, 121, 238): 32,
        (128, 0, 255): 33,
        (255, 0, 255): 34,
        (135, 206, 255): 35,
        (241, 230, 255): 36,
        (96, 69, 143): 37,
        (53, 46, 82): 38
    }  
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images
        self.masks_fps = labels
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def create_instance_labels(self, label_image): 
        mask = np.zeros((label_image.shape[0], label_image.shape[1]))

        for key in self.INSTANCE_LABELS.keys(): 
            red_img = np.logical_and(np.logical_and((label_image[:,:,0] == key[0]), (label_image[:,:,1] == key[1])), (label_image[:,:,2] == key[2])) 
            mask[red_img] = self.INSTANCE_LABELS[key]

        return mask
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(self.masks_fps[i])
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        mask = self.create_instance_labels(label_img) 

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    
    def visualize(self, i): 
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image 