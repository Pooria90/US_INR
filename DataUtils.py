'''
This modules changes the Zenodo data file to an annotation file (name vs label)
for using to make the Pytorch dataset.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import keras_ocr
import cv2
import math
import torch
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


'''
Credits for Inpaiting section:
Carlo Borella at https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4

'''
#  Inpaiting to remove text from images
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

'''
def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(img)
'''

# This function generates a meshgrid based on the size of a square image
def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

# A function to resize all the images in a directory prior to the training process
# This makes the training much faster if we are to deal with a fixed set of images
def resize_images(path, dest, size = (224,224)):
    '''
    path: images directory
    dest: destination folder
    size: final size of the images
    '''
    dirs = os.listdir(path)

    for ii, item in enumerate(dirs):
        image = np.asarray(Image.open(path + item).convert("L"))
        print(f'Image {ii+1} shape: {image.shape}')
        imResize = cv2.resize(image, size)
        cv2.imwrite(dest + item, imResize)
        print(f'Image {ii+1} Done!')


# A torch dataset for working with normal US images
class US_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device='cpu'):
        self.img_labels = pd.read_csv(annotations_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY).to(self.device) / 255
        label = torch.tensor(self.img_labels.iloc[idx, 1]).to(self.device)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Torch dataset for working with INRs
class INR_Dataset(Dataset):
    def __init__(self, data_table, img_dir, size = 224, device='cpu'):
        super().__init__()
        self.data = data_table
        self.img_dir = img_dir
        #self.transform = transform # in case of other transforms
        self.size = size
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.data.iloc[idx, 0] + '.png'
        image = read_image(img_path, mode=ImageReadMode.GRAY).to(self.device) / 255

        transform = Resize((self.size, self.size), antialias=True)
        image = transform(image)

        pixels = image.permute(1, 2, 0).view(-1, 1)
        coords = get_mgrid(self.size, 2).to(self.device)
        return coords, pixels, image
    

if __name__ == "__main__":
    import pandas as pd

    labels = {
        'Fetal abdomen': 0,
        'Trans-thalamic': 1,
        'Trans-cerebellum': 2,
        'Trans-ventricular': 3,
        'Fetal femur': 4,
        'Fetal thorax': 5,
        'Maternal cervix': 6,
        'Other': 7
    }

    data_table = pd.read_csv('Data/FETAL_PLANES_DB_data.csv', sep=';')
    brain_dropped = data_table[data_table['Plane'] != 'Fetal brain'].loc[:, ['Image_name', 'Plane']]
    brain_data = data_table[data_table['Plane'] == 'Fetal brain'].loc[:, ['Image_name', 'Brain_plane']]
    brain_data = brain_data.set_axis(['Image_name', 'Plane'], axis=1)
    annotations = pd.concat([brain_dropped, brain_data], ignore_index=True)
    annotations = annotations.replace(to_replace=labels)
    annotations['Image_name'] += '.png'
    
    #print (brain_dropped)
    #print (brain_data)
    #print (pd.concat([brain_dropped, brain_data], ignore_index=True))
    #print (annotations)

    annotations.to_csv(path_or_buf='Data/annotations.csv', sep=';', index=False)