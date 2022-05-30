import os 
import matplotlib.pyplot as plt 

def list_dir_recursive(input_path, endswith=None):
    '''
    Find all files in a given folder, in a recursive manner. 
    Returns the list of all file paths. 
    
    Optional: add a filetype ending. e.g. "png" 
    '''
    image_files = []
    for root, dirs, files in os.walk(input_path, topdown=True):
        for f in files: 
            if not endswith or f.endswith(endswith):
                image_files.append(os.path.join(root, f))
    return image_files




def visualize(**images):
    """
    Plot image, ground truth and prediction in one row.
    Code from https://github.com/qubvel/segmentation_models.pytorch ! 
    """
    
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
    

def check_cityscapes_labels_found(images, labels): 
    """
    For CityScapes it is rather easy, as each image has a corresponding label, no errors will be found.
    If some file is missing it will print out which sample is missing. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('images/')[-1]
        lb = lbl.split('labels/')[-1]
        
        if not im.replace('_leftImg8bit.png', '') == lb.replace('_gtFine_labelIds.png', ''):
            print('Missmatch here\n {}\n {}: '.format(im, lb))
            return
    print('All labels found')

    

def check_bdd100k_images_found(images, labels): 
    """
    For CityScapes it is rather easy, as each image has a corresponding label, no errors will be found.
    If some file is missing it will print out which sample is missing. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('images/')[-1]
        lb = lbl.split('labels/')[-1]
        
        im = im.replace('.jpg', '')
        lb = lb.replace('-mask.png', '') 
        
        if not im == lb:
            print('Missmatch here\n {}\n {}: '.format(im, lb))
            return
    print('All labels found')