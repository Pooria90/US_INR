'''
A file to train the Functa network.
'''

import os
from copy import deepcopy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from Models import ModulatedSineLayer, ModulatedSiren
from DataPrep import get_mgrid, INR_Dataset

# Training code
def train_functa(
        model,
        train_ds,
        valid_ds,
        num_iter,
        bs,
        N_inner,
        lr_outer,
        lr_inner,
    ):

    model.train()

    meta_optimizer = torch.optim.Adam(lr=lr_outer, params=model.parameters())
    # === Maybe add a scheduler === #

    # === initialize some logger in here === #

    meta_grad_init = [0 for _ in range(len(model.state_dict()))] # starting point for meta-gradients

    # === Still don't know how to make train_ds and valid_ds === #
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True) # === Check CAVIA for pin_memory
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True) # === Check CAVIA for pin_memory

    for iter in range(1, num_iter + 1):
        # === Some logs === #

        for counter, (xb,yb,imgs) in enumerate(train_dl):
            if xb.shape[0] != bs:
                continue

            meta_grad = deepcopy(meta_grad_init)

            # === Some logs === #

            for j in range(bs):
                model.reset_modulation()
                # --- inner update
                for i in range(N_inner):
                    # prediction
                    pred_train = model(xb[j])

                    # loss
                    loss_train = F.mse_loss(pred_train, yb[j])

                    # grad, There are some note in the CAVIA's supplemetary matrial
                    grad_train = torch.autograd.grad(loss_train, model.modulation, create_graph=True)[0]

                    # update modulations
                    model.modulation = model.modulation - lr_inner * grad_train

                    #print(f'iter: {iter} --- batch: {counter+1} --- sample: {j+1} --- inner: {i+1}')
                    #print(f'loss train = {loss_train}')

                # --- meta-gradients
                pred_test = model(xb[j])
                loss_test = F.mse_loss(pred_test, yb[j])
                grad_test = torch.autograd.grad(loss_test, model.parameters())
                for i in range(len(grad_test)):
                    meta_grad[i] += grad_test[i].detach()

                print(f'Iter: {iter} --- batch: {counter+1} --- loss test for meta-training: {loss_test}')
                # === Some logs === #

            model.reset_modulation()

            # === Some logs to summarize inner loop === #

            # === Evaluation and saving the checkpoint === #

            # --- Meta-update
            meta_optimizer.zero_grad()

            # setting gradients
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(bs)
                param.grad.data.clamp_(-10, 10) # based on CAVIA

            meta_optimizer.step()

    model.reset_modulation()
    return model # and probably log data


# Running the main process
if __name__ == '__main__':
    # === Arguments here === #

    # === Seeds here === #
    seed = 216

    # === Changing directory
    os.chdir('./Data')
    print ('Current working directory: ' + os.getcwd() + '\n')

    # === Reading the data table
    data_table = pd.read_csv('FETAL_PLANES_DB_data.csv', sep=';')
    print (data_table.tail(10))

    # === Extracting the required data
    brain_data = data_table[data_table['Plane'] == 'Fetal brain']
    brain_data = brain_data.reset_index(drop = True)
    print (brain_data.tail(5))

    # === Data prepration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = INR_Dataset(brain_data, './Images/', 144, device=device)

    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(ds, [0.05, 0.95], generator=generator)
    print ('\nTrain size: ',train_ds.__len__(), '--- Valid size: ', valid_ds.__len__())

    # === Model setup
    model = ModulatedSiren(
        in_features=2,
        hidden_features=[32,32],
        num_modulations=16,
        out_features=1,
        last_linear=True,
        device = device,
        first_omega_0=200,
        hidden_omega_0=200
    ).to(device)
    
    summary(model, (2,), device = device)
    
    # === Model training
    model = train_functa(model, train_ds, valid_ds, 3, 8, 2, 5e-6, 0.001)


