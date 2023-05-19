'''
This file is intended to run and test the modules.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from copy import deepcopy
from datetime import datetime
import time

import DataPrep
import Models
import Fitting

device = 'cuda'

# --------------------------------------------- Reading the data
img_dir = 'Data/Images/'
anot_dir = 'Data/annotations.csv'
resize = Resize(size=(224,224), antialias=True)
fetal_planes = DataPrep.US_Dataset(annotations_file=anot_dir, img_dir=img_dir, transform=resize, device=device)

# --------------------------------------------- Splitting the dataset
generator = torch.Generator().manual_seed(216)
train_ds, valid_ds, test_ds = random_split(fetal_planes, [0.75, 0.15, 0.1], generator=generator)
print(f'Train size: {train_ds.__len__()}\nValid size: {valid_ds.__len__()}\nTest size: {test_ds.__len__()}\n')

# --------------------------------------------- Visualization
flag = False

if flag:
    m, n = 5, 5
    index = torch.randint(train_ds.__len__(), (m,n))
    fig, ax = plt.subplots(m,n, figsize = (15,15))
    for i in range(m):
        for j in range(n):
            img, label = train_ds.__getitem__(index[i,j])
            ax[i,j].imshow(torch.squeeze(img).cpu().numpy(), cmap='gray')
            
# --------------------------------------------- Training the model
model = Models.SonoNet().to(device)

now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S-")
save_path = 'Results/' + date_time + model.name
os.mkdir(save_path)

start = time.time()
hist = Fitting.train(model, train_ds, valid_ds, batch_size=372, epochs=10, learning_rate=0.001, period=1, save_path=save_path)
end = time.time()

print(f'The whole training process time: {end-start}')
