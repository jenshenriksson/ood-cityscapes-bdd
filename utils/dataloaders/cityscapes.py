import cv2 
import numpy as np
from torch.utils.data import Dataset as Dataset

import sys
sys.path.append('..')
import time

from tqdm.notebook import tqdm
import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize


HEIGHT = 384
WIDTH = 768

albuHeight = albu.Compose([
    albu.LongestMaxSize(max_size=WIDTH, always_apply=True), 
    albu.CenterCrop(height=HEIGHT, width=WIDTH, always_apply=True),
])
albuWidth = albu.Compose([
    albu.SmallestMaxSize(max_size=HEIGHT, always_apply=True),
    albu.CenterCrop(height=HEIGHT, width=WIDTH, always_apply=True),
])



class CityScapes(Dataset):
    """
    CityScapes Dataset. Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available here: https://www.cityscapes-dataset.com/dataset-overview/
    
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
    
    CLASSES = [
        'unlabeled', 
        'ego vehicle',
        'rectification border', 
        'out of roi', 
        'static', 
        'dynamic', 
        'ground', 
        'road', 
        'sidewalk', 
        'parking', 
        'rail track',  
        'building', 
        'wall', 
        'fence', 
        'guard rail', 
        'bridge', 
        'tunnel',  
        'pole', 
        'polegroup', 
        'traffic light', 
        'traffic sign',  
        'vegetation', 
        'terrain',  
        'sky',  
        'person', 
        'rider',  
        'car', 
        'truck', 
        'bus', 
        'caravan',
        'trailer', 
        'train', 
        'motorcycle', 
        'bicycle',   
        "garage",                  # Added for KITTI-360 (Used to be Licence Plate for CityScapes)
        "gate",                    # Added for KITTI-360
        "stop",                    # Added for KITTI-360
        "smallpole",               # Added for KITTI-360
        "lamp",                    # Added for KITTI-360
        "trash bin",               # Added for KITTI-360
        "vending machine",         # Added for KITTI-360
        "box",                     # Added for KITTI-360
        "unknown construction",    # Added for KITTI-360 
        "unknown vehicle",         # Added for KITTI-360
        "unknown object",          # Added for KITTI-360
        "license plate"            # Added for KITTI-360
    ]
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            preload=False, 
    ):
        self.images_fps = images
        self.masks_fps = labels
        self.preload = preload 
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
       
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
                
        # Store images before dataloader. 
        if self.preload: 
            self.preloadedImages = []
            self.preloadedMasks = [] 
            
            for i in tqdm(range(len(self.images_fps))):
                image = cv2.imread(self.images_fps[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.masks_fps[i], 0)
                
                # Resize
                if image.shape[1] / image.shape[0] >= 2: 
                    image = albuWidth(image=image)['image']
                    mask = albuWidth(image=mask)['image']
                else: 
                    image = albuHeight(image=image)['image']
                    mask = albuHeight(image=mask)['image']
                
                self.preloadedImages.append(image)
                self.preloadedMasks.append(mask) 
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
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


class CityScapes_class_merge(Dataset):
    """
    Adjusted CityScapes/KITTI360 dataloader, that converts the classes as described in Notebook 2. 
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available here: https://www.cityscapes-dataset.com/dataset-overview/
    
    
    Adjust the labels for the following classes, as described in Notebook 2. 
    Move guard rail to fence class.
    Move bridge and tunnel to buildings class.
    Move polegroup to pole class.
    Move caravan, trailer and license plate to car.
    Move parking to road, as we consider it an area where the vehicle may drive/park.
    Move ground to terrain, as it is part of the area where a vehicle is not supposed to drive.
    Move rail track to terrain, as it is part of the area where a vehicle is not supposed to drive.

    
    
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
    
    CLASSES = [
        'unlabeled', 
        'ego vehicle',
        'rectification border', 
        'out of roi', 
        'static', 
        'dynamic', 
        'ground', 
        'road', 
        'sidewalk', 
        'parking', 
        'rail track',  
        'building', 
        'wall', 
        'fence', 
        'guard rail', 
        'bridge', 
        'tunnel',  
        'pole', 
        'polegroup', 
        'traffic light', 
        'traffic sign',  
        'vegetation', 
        'terrain',  
        'sky',  
        'person', 
        'rider',  
        'car', 
        'truck', 
        'bus', 
        'caravan',
        'trailer', 
        'train', 
        'motorcycle', 
        'bicycle', 
        'license plate',  
    ]
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            preload=False, 
    ):
        self.images_fps = images
        self.masks_fps = labels
        self.preload = preload 
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
        
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = list()
        for cls in classes:
            if cls == 'fence': 
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('guard rail')])
            elif cls == 'building':
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('bridge'), self.CLASSES.index('tunnel')])
            elif cls == 'pole': 
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('polegroup')])
            elif cls == 'car': 
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('caravan'), self.CLASSES.index('trailer'), self.CLASSES.index('license plate')])
            elif cls == 'road': 
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('parking')])
            elif cls == 'terrain': 
                self.class_values.append([self.CLASSES.index(cls.lower()), self.CLASSES.index('ground'), self.CLASSES.index('rail track')])
            elif cls in ['guard rail', 'bridge', 'tunnel', 'polegroup', 'caravan', 'trailer', 'license plate', 'parking', 'ground', 'rail track']:
                continue 
            else: 
                self.class_values.append(self.CLASSES.index(cls.lower()))
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
                
        # Store images before dataloader. 
        if self.preload: 
            self.preloadedImages = []
            self.preloadedMasks = [] 
            
            for i in tqdm(range(len(self.images_fps))):
                image = cv2.imread(self.images_fps[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.masks_fps[i], 0)
                
                # Resize
                if image.shape[1] / image.shape[0] >= 2: 
                    image = albuWidth(image=image)['image']
                    mask = albuWidth(image=mask)['image']
                else: 
                    image = albuHeight(image=image)['image']
                    mask = albuHeight(image=mask)['image']
                
                
                self.preloadedImages.append(image)
                self.preloadedMasks.append(mask) 
        
    
    def __getitem__(self, i):
        
        # read data
        if not self.preload: 
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
        else: 
            image = self.preloadedImages[i]
            mask = self.preloadedMasks[i] 
        
        # extract certain classes from mask (e.g. cars)
        masks = list()
        for cls_value in self.class_values: 
            if isinstance(cls_value, list):  # If cls is a list, i.e. modified from above. 
                temp = [(mask == v) for v in cls_value]
                temp_numpy = np.stack(temp, axis=0).astype('float')
                # temp_mask = temp_numpy
                temp_mask = np.logical_or.reduce(temp_numpy, axis=0)*1  # Merge all masks for the given class. 
            else: 
                temp_mask = (mask == cls_value)
            masks.append(temp_mask) 
        
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