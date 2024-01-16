'''
This file containes the classes for logging during INR training.
'''
import os
import string, random
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from datetime import datetime
from copy import deepcopy

def present_time():
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    return date_time

class Logger():
    def __init__(self, log_period, verbose, args = None):
        self.log_period = log_period
        self.args = args
        self.verbose = verbose
        self.start_time = time()

        self.best_train_loss = np.inf
        self.best_valid_loss = np.inf

        self.best_model_train = None
        self.best_model_valid = None

        self.best_model_train_stats = None
        self.best_model_valid_stats = None

        self.best_optim_train = None
        self.best_optim_valid = None

        # === if verbose, print something ===

        self.train_stats = {
            'train_loss_pre_update': [],
            'train_loss_post_update': []
        }

        self.valid_stats = {
            'valid_loss_pre_update': [],
            'valid_loss_post_update': []
        }

        t = present_time()
        if not os.path.exists('Results/'):
            os.mkdir('Results/')
        self.path = 'Results/Logs ' + t + '/'
        # The following a if-statement is to avoid a rare case where an HPC server (UBC ARC Sockeye in my case) starts to jobs at the same time
        if os.path.exists(self.path):
            random_str = ''.join(random.choices(string.ascii_lowercase,k=2))
            self.path = 'Results/Logs ' + t + f'-{random_str}' + '/'
        # --------------------------------------------------------------------
        os.mkdir(self.path)
        if self.args != None:
            with open(self.path + 'args.txt', 'w') as f:
                f.writelines(self.args)
        print (f'Logger intialized at {t}!')

    def prepare_inner_loop(self, iter, mode='train'):
        # Called before iterating over the batch in the inner loop

        if iter % self.log_period == 0:
            if mode == 'train':
                for key in self.train_stats.keys():
                    self.train_stats[key].append([])
            elif mode == 'valid':
                for key in self.valid_stats.keys():
                    self.valid_stats[key].append([])
            else:
                raise NotImplementedError()

    def log_pre_update(self, iter, x, y, model, mode='train'):
        if iter % self.log_period == 0:
            if mode == 'train':
                self.train_stats['train_loss_pre_update'][-1].append(self.get_loss(x, y, model))
            elif mode == 'valid':
                self.valid_stats['valid_loss_pre_update'][-1].append(self.get_loss(x, y, model))
            else:
                raise NotImplementedError()

    def log_post_update(self, iter, x, y, model, mode='train'):
        if iter % self.log_period == 0:
            if mode == 'train':
                self.train_stats['train_loss_post_update'][-1].append(self.get_loss(x, y, model))
            elif mode == 'valid':
                self.valid_stats['valid_loss_post_update'][-1].append(self.get_loss(x, y, model))
            else:
                raise NotImplementedError()

    def summarise_inner_loop(self, iter, mode):
        if iter % self.log_period == 0:
            if mode == 'train':
                for key in self.train_stats.keys():
                    self.train_stats[key][-1] = np.mean(self.train_stats[key][-1])
            if mode == 'valid':
                for key in self.valid_stats.keys():
                    self.valid_stats[key][-1] = np.mean(self.valid_stats[key][-1])

    def get_loss(self, x, y, model):
        pred = model(x)
        return F.mse_loss(pred, y).item()

    def print_logs(self, iter, grad_inner, grad_meta):
        if self.verbose and iter % self.log_period == 0:
            print(f'*** Epoch {iter} ***')
            print('Train loss: {} -> {}'.format(
                self.train_stats['train_loss_pre_update'][-1],
                self.train_stats['train_loss_post_update'][-1]
            ))
            print('Valid loss: {} -> {}'.format(
                self.valid_stats['valid_loss_pre_update'][-1],
                self.valid_stats['valid_loss_post_update'][-1]
            ))
            print('Inner grad: {}'.format(
                grad_inner[0].abs().mean().item()
            ))
            print('Meta  grad: {}'.format(
                grad_meta[0].abs().mean().item()
            ))
            print('Time elaps: {:.2f} mins'.format(
                (time() - self.start_time)/60
            ))

    def update_best_model(
        self,
        iter,
        logger,
        model,
        optim,
        save_path = None
    ):

        if save_path == None:
            save_path = self.path

        tr_loss = self.train_stats['train_loss_post_update'][-1]
        va_loss = self.valid_stats['valid_loss_post_update'][-1]

        if tr_loss < self.best_train_loss:
            self.best_train_loss = tr_loss
            self.best_model_train = deepcopy(model)
            self.best_optim_train = deepcopy(optim)
            self.best_model_train_stats = {
                'tr_loss': tr_loss,
                'va_loss': va_loss,
                'epoch': iter
            }
            np.save(save_path + 'best_model_train_stats.npy', self.best_model_train_stats)
            torch.save(deepcopy(self.best_model_train).to('cpu'), save_path + 'best_model_train')
            torch.save(self.best_optim_train, save_path + 'best_optim_train')

        if va_loss < self.best_valid_loss:
            self.best_valid_loss = va_loss
            self.best_model_valid = deepcopy(model)
            self.best_optim_valid = deepcopy(optim)
            self.best_model_valid_stats = {
                'tr_loss': tr_loss,
                'va_loss': va_loss,
                'epoch': iter
            }
            np.save(save_path + 'best_model_valid_stats.npy', self.best_model_valid_stats)
            torch.save(deepcopy(self.best_model_valid).to('cpu'), save_path + 'best_model_valid')
            torch.save(self.best_optim_valid, save_path + 'best_optim_valid')

    def save_logger(self, path):
        pass

    def save_checkpoint(self, path):
        pass
