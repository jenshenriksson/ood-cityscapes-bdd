import torch
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_iou(prediction, gt_mask, eps=1e-7, th_mask=None):  
    '''
    pd_mask: Prediction mask 
    gt_mask: Ground truth mask
    th_mask: Threshold mask 
    eps: epsilon to avoid zero division
    '''
    if th_mask is not None: 
        pr_mask = prediction > th_mask
    else: 
        pr_mask = (prediction.round())
    
    area_of_overlap = np.logical_and(gt_mask, pr_mask).sum()
    area_of_union = np.logical_or(gt_mask, pr_mask).sum() + eps 
    intersection_over_union = area_of_overlap / area_of_union

    return intersection_over_union


def compute_class_pixel_distribution(dataset, device=None, dims=(0, 1, 2)): 
    '''
    The compute_class_pixel_distribution function computes the pixel-wise distribution of classes in a given dataset. 
    
    It takes three arguments:
    dataset: a torch.utils.data.Dataset object containing the dataset to compute the distribution for.
    device: (optional) the device to perform computations on. If not provided, the computations will be performed on the CPU.
    dims: (optional) a tuple of dimensions to sum over when computing the distribution. 
    '''
    num_classes = len(dataset.class_values)
    samples_per_class = torch.zeros(num_classes).to(device)
    total_samples = torch.zeros(1).to(device)
    total_annotated = torch.zeros(1).to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    itr = tqdm(loader, total=len(loader))
    
    with torch.no_grad(): 
        for _, lbl in itr: 
            lbl = lbl.to(device)
            this_frame = lbl.sum(dim=dims) 
            samples_per_class += this_frame / (lbl.numel() / num_classes) 
            total_annotated += lbl.sum() / (lbl.numel() / num_classes)
            total_samples += lbl.size(0)
  
    return samples_per_class, total_samples, total_annotated


def iou_test_per_class(loader, model, num_classes, device, use_sigmoid=False, dims=(0, 2, 3), eps=1):
    '''The iou_test_per_class function calculates the intersection over union (IOU) metric for each 
    class in a dataset using a given model.

    It takes three arguments:
    loader: A data loader object that yields batches of images and labels
    model: A model object that takes images as input and outputs predictions
    num_classes: An integer representing the number of classes in the dataset
    device: The device (e.g. "cpu" or "cuda") on which the tensors should be stored
    use_sigmoid: A boolean flag indicating whether to apply the sigmoid function to the model's output (default: False)
    dims: A tuple of dimensions along which to sum the logical AND and logical OR operations (default: (0, 2, 3))
    eps: A small constant to add to the denominator of the IOU calculation to avoid division by zero (default: 1)
    
    The iou_test_per_class function returns a tuple containing the following: 
    intersection_per_class: A tensor of shape (num_classes,) representing the sum of intersections per class
    total_union_per_class: A tensor of shape (num_classes,) representing the sum of unions per class
    '''
    model.eval()
    intersection_per_class = torch.zeros(num_classes).to(device)
    total_union_per_class = torch.zeros(num_classes).to(device)
    ctr = 0 
    with torch.no_grad():
        itr = tqdm(loader, total=len(loader))
        for images, labels in itr: 
            images = images.to(device)
            labels = labels.to(device)

            if use_sigmoid: 
                preds = torch.sigmoid(model(images))
            else: 
                preds = model(images)
                
            pd_mask = (preds > 0.5).float() 
            intersection_per_class += (torch.logical_and(pd_mask, labels)).sum(dim=dims)  # Reduces away dims, so only dim1 remains
            total_union_per_class += (torch.logical_or(pd_mask, labels)).sum(dim=dims)
            itr.set_postfix({'avg iou': intersection_per_class.sum()/(total_union_per_class.sum()+eps)})    
        
    return intersection_per_class, total_union_per_class


def class_existence(dataset, device=None, dims=(0, 1, 2)): 
    '''
    The class_existence function calculates the number of pixels belonging to each class for each image in a given dataset. 
    It does this by creating a tensor of shape (num_images, num_classes) that is initialized to zeros, 
    and then iterating over the dataset in a data loader, summing the number of pixels belonging 
    to each class for each image, and storing the result in the tensor. 
    The function takes the following inputs:

    dataset: A dataset object containing images and labels
    device: The device (e.g. "cpu" or "cuda") on which the tensors should be stored (default: None)
    dims: A tuple of dimensions along which to sum the pixels belonging to each class (default: (0, 1, 2))
    
    The class_existence function returns a tensor of shape (num_images, num_classes) 
    representing the number of pixels belonging to each class for each image in the dataset.
    '''
    num_classes = len(dataset.class_values)
    num_images = len(dataset)
    
    samples_per_image = torch.zeros(num_images, num_classes).to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    itr = tqdm(loader, total=len(loader))
    
    with torch.no_grad(): 
        for idx, (_, lbl) in enumerate(itr): 
            lbl = lbl.to(device)
            samples_per_image[idx] += lbl.sum(dim=dims)
    return samples_per_image