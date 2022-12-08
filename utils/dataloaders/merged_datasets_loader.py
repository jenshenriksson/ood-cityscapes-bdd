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


    

class ListOfDatasetsLoader(Dataset):
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

    def __init__(self, datasets, shuffle=False):
        self.datasets = datasets
        
        previous_length = 0
        self.dataset_sizes = []
        for ds in self.datasets: 
            if ds:  # If DS is not empty 
                dataset_len = len(ds)
                previous_length += dataset_len 
                self.dataset_sizes.append(previous_length)
        
        self.sample_indexes = np.arange(0, previous_length).astype('int')
        self.shuffle = shuffle 
        if self.shuffle: 
            self.shuffle_samples() 
    
    def shuffle_samples(self): 
        np.random.shuffle(self.sample_indexes)
        
    def __len__(self): 
        return(np.max(self.dataset_sizes))
    
    def __getitem__(self, i):
        if i == 0 and self.shuffle:  # New epoch 
            self.shuffle_samples() 
        idx = self.sample_indexes[i]
        
        offset = 0 
        for ds, val in enumerate(self.dataset_sizes): 
            if idx < val:
                return self.datasets[ds][idx-offset]
            offset += val

    def visualize(self, i):
        
        idx = self.sample_indexes[i]
        
        offset = 0 
        for ds, val in enumerate(self.dataset_sizes): 
            if idx < val:
                return self.datasets[ds].visualize(idx-offset)
            offset += val
            
