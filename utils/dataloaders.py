import cv2 
import numpy as np
from torch.utils.data import Dataset as Dataset
# from parameters import * 
import sys
sys.path.append('..')
import utils as CU
import time

from tqdm.notebook import tqdm
import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize

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
        'license plate',  
    ]
    
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
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

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
    
    
class BDD100K(Dataset):
    """    
    DataLoader for the Semantic Segmentation subset of BDD100K. 
    Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
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
        'road', 
        'sidewalk', 
        'building', 
        'wall', 
        'fence',  
        'pole', 
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
        'train', 
        'motorcycle', 
        'bicycle',  
        'unknown', 
    ] 
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            preload = False, 
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
            
            albuResize = albu.Compose([
                Resize(height=256, width=512)
            ])
            for i in tqdm(range(len(self.images_fps))):
                image = cv2.imread(self.images_fps[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.masks_fps[i], 0)
                
                # Resize
                image = albuResize(image=image)['image']
                mask = albuResize(image=mask)['image']
                
                self.preloadedImages.append(image)
                self.preloadedMasks.append(mask) 
    
    def __getitem__(self, i):
        if not self.preload: 
            # read data
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
        else: 
            image = self.preloadedImages[i]
            mask = self.preloadedMasks[i]

        # extract certain classes from mask + convert to one-hot. 
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
    

class CityScapes_class_merge(Dataset):
    """
    Adjusted CityScapes dataloader, that converts the classes as described in Notebook 2. 
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
            
            albuResize = albu.Compose([
                Resize(height=256, width=512)
            ])
            for i in tqdm(range(len(self.images_fps))):
                image = cv2.imread(self.images_fps[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.masks_fps[i], 0)
                
                # Resize
                image = albuResize(image=image)['image']
                mask = albuResize(image=mask)['image']
                
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
    

class CityScapes_bdd100k_merge(Dataset):
    ''' 
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
    '''
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 
        'fence', 'pole', 'traffic light', 'traffic sign', 
        'vegetation', 'terrain', 'sky', 
        'person', 'rider', 
        'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    def __init__(self, cityscapes, bdd100k, shuffle=False):
        self.cityscapes = cityscapes 
        self.bdd100k = bdd100k
        
        self.bdd_len = len(bdd100k)
        self.cityscapes_len = len(cityscapes)
        self.shuffle = shuffle 
        
        self.bdd_arange = np.arange(0, self.bdd_len)
        self.cityscapes_arange = np.arange(self.bdd_len, self.bdd_len+self.cityscapes_len)
        self.sample_indexes = np.concatenate([self.bdd_arange, self.cityscapes_arange]) 
        if self.shuffle: 
            self.shuffle_samples() 
     
    def ignore_pre_processor(self):
        # For image visualization: Ignore the preprocessor for a separate dataloader. 
        self.cityscapes.preprocessing = None
        self.bdd100k.preprocessing = None 
    
    def shuffle_samples(self): 
        np.random.shuffle(self.sample_indexes)
        
    def __getitem__(self, i):
        if i == 0 and self.shuffle:  # New epoch 
            self.shuffle_samples() 
        idx = self.sample_indexes[i]
        
        if idx < self.bdd_len:
            return self.bdd100k[idx]
        return self.cityscapes[idx-self.bdd_len] 
        
    def __len__(self): 
        return(self.bdd_len + self.cityscapes_len)
    
    def visualize(self, i):
        idx = self.sample_indexes[i]
        if idx < self.bdd_len:
            return self.bdd100k.visualize(idx)
        return self.cityscapes.visualize(idx-self.bdd_len)
