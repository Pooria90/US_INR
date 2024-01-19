'''
A file to train the Functa network.
It uses typer options for better compability with HPC systems.
'''

import os
import typer
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

from models import ModulatedSineLayer, ModulatedSiren
from datautils import get_mgrid, INR_Dataset
from logging import Logger, present_time

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
        lr_meta_decay = 0.9,
        meta_optimizer = None,
        ep_start = None,
        log_period = 1,
        verbose = True,
        args = None,
        loss_func = F.mse_loss
    ):

    model.train()

    if meta_optimizer == None:
        meta_optimizer = torch.optim.Adam(lr=lr_outer, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, 5000, lr_meta_decay)

    # logs intialization
    print ('===> Training started <===\n')
    #print (f'Date and time: {present_time()}')
    logger = Logger(log_period=log_period, verbose=verbose, args=args)

    meta_grad_init = [0 for _ in range(len(model.state_dict()))] # starting point for meta-gradients

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True) # === Check CAVIA for pin_memory
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True) # === Check CAVIA for pin_memory

    if ep_start == None:
        ep_start = 1

    iter = ep_start
    while iter <= num_iter:
        for counter, (xb,yb,imgs) in enumerate(train_dl):
            if xb.shape[0] != bs:
                continue

            meta_grad = deepcopy(meta_grad_init)

            logger.prepare_inner_loop(iter)

            for j in range(bs):
                model.reset_modulation()

                logger.log_pre_update(iter, xb[j], yb[j], model)

                # --- inner update
                for i in range(N_inner):
                    # prediction
                    pred_train = model(xb[j])

                    # loss
                    loss_train = loss_func(pred_train, yb[j])

                    # grad, There are some note in the CAVIA's supplemetary matrial
                    grad_train = torch.autograd.grad(loss_train, model.modulation, create_graph=True)[0]

                    # update modulations
                    model.modulation = model.modulation - lr_inner * grad_train

                    #print(f'iter: {iter} --- batch: {counter+1} --- sample: {j+1} --- inner: {i+1}')
                    #print(f'loss train = {loss_train}')

                # --- meta-gradients
                pred_test = model(xb[j])
                loss_test = loss_func(pred_test, yb[j])
                grad_test = torch.autograd.grad(loss_test, model.parameters())
                for i in range(len(grad_test)):
                    meta_grad[i] += grad_test[i].detach()

                # print(f'loss test for meta-training: {loss_test}')
                logger.log_post_update(iter, xb[j], yb[j], model)

            model.reset_modulation()

            logger.summarise_inner_loop(iter, mode='train')

            if iter % log_period == 0:
                evaluate(iter, model, logger, valid_dl, N_inner, lr_inner)
                logger.update_best_model(iter, logger, model, meta_optimizer)
            logger.print_logs(iter, grad_train, meta_grad)
            # === save checkpoints and stats ===

            # --- Meta-update
            meta_optimizer.zero_grad()

            # setting gradients
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(bs)
                param.grad.data.clamp_(-10, 10) # based on CAVIA

            meta_optimizer.step()
            scheduler.step()
            
            iter += 1
            if iter > num_iter:
                break

    model.reset_modulation()
    return logger, model

# Evaluation function
def evaluate(
        iter,
        model,
        logger,
        dataloader,
        N_inner,
        lr_inner,
        loss_func = F.mse_loss
    ):
    logger.prepare_inner_loop(iter, mode='valid')

    for counter, (xb,yb,_) in enumerate(dataloader):
        for j in range(xb.shape[0]):

            model.reset_modulation()

            # --- inner update
            logger.log_pre_update(iter, xb[j], yb[j], model, mode='valid')

            for _ in range(N_inner):
                # prediction
                pred_train = model(xb[j])

                # loss
                loss_train = loss_func(pred_train, yb[j])

                # grad, There are some note in the CAVIA's supplemetary matrial
                grad_train = torch.autograd.grad(loss_train, model.modulation, create_graph=True)[0]

                # update modulations
                model.modulation = model.modulation - lr_inner * grad_train

            logger.log_post_update(iter, xb[j], yb[j], model, mode='valid')

    # reset context parameters
    model.reset_modulation()

    # this will take the mean over the batches
    logger.summarise_inner_loop(iter, mode='valid')


def main_process(
        seed: int = 216,
        data_path: str = './Data/Images/', #Now, datapath should directly point to the Images folder.
        table_path: str = './Data/data_table.csv',
        image_len: int = 128,
        valid_split: float = 0.2,
        in_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 10,
        num_modulations: int = 256,
        out_features: int = 1,
        last_linear: bool = True,
        first_omega_0: int = 100,
        hidden_omega_0: int = 100,
        num_iter: int = 10000,
        batch_size: int = 8,
        N_inner: int = 2,
        lr_outer: float = 5e-5,
        lr_inner: float = 0.01,
        ep_start: int = 1,
        log_period: int = 20
):
    
    args_str = '''
        seed: {},
        data_path: {},
        table_path: {},
        image_len: {},
        valid_split: {},
        in_features: {},
        hidden_features: {},
        hidden_layers: {},
        num_modulations: {},
        out_features: {},
        last_linear: {},
        first_omega_0: {},
        hidden_omega_0: {},
        num_iter: {},
        batch_size: {},
        N_inner: {},
        lr_outer: {},
        lr_inner: {},
        ep_start: {},
        log_period: {}
    '''.format(
        seed,data_path,table_path,image_len,valid_split,in_features,hidden_features,
        hidden_layers,num_modulations,out_features,last_linear,
        first_omega_0,hidden_omega_0,num_iter,batch_size,N_inner,
        lr_outer,lr_inner,ep_start,log_period
    )
    
    print (args_str)
        
    #os.chdir(data_path)
    print ('Current working directory: ' + os.getcwd() + '\n')

    # === Reading the data table; data_table.csv is the metadata file which is not created yet.
    data_table = pd.read_csv(table_path, sep=';')
    print (data_table.tail(10))

    # === Extracting the required data; This part needs to be deleted. Data should be chosen outside!
    # Also, metadata should have the name of images including their format (.png, .jpg, ...) in the first column.
    '''brain_data = data_table[data_table['Plane'] == 'Fetal brain']
    brain_data = brain_data.reset_index(drop = True)
    print (brain_data.tail(5))'''

    # === Data prepration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = INR_Dataset(data_table, data_path, image_len, device=device)

    # === Data Split
    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(ds, [1-valid_split, valid_split], generator=generator)
    print ('\nTrain size: ',train_ds.__len__(), '--- Valid size: ', valid_ds.__len__())

    # === Model setup
    model = ModulatedSiren(
        in_features=in_features,
        hidden_features=[hidden_features]*hidden_layers,
        num_modulations=num_modulations,
        out_features=out_features,
        last_linear=last_linear,
        device = device,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0
    ).to(device)
    
    summary(model, (in_features,), device = device)
    
    # === Model training
    logger, model = train_functa(model, train_ds, valid_ds, num_iter, batch_size, N_inner, lr_outer, lr_inner, ep_start = ep_start, log_period = log_period, args = args_str)
    
    return


# Running the main process
if __name__ == '__main__':
    typer.run(main_process)
