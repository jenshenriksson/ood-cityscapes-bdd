import cv2 
import numpy as np
from torch.utils.data import Dataset as Dataset



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
        'wall', 'fence', 
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