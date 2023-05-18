'''
This modules changes the Zenodo data file to an annotation file (name vs label)
for using to make the Pytorch dataset.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

