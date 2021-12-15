import torch 
import numpy as np
import validate
import config
import load_data
import run
from torch.utils.data import Dataset, DataLoader, TensorDataset

#train the data
#data_index 0 = vase, 1= plane
#iter is the number of iterations
#train_mode = 0: only trains on original images
#train_mode = 1: only trains on augmented images
#train_mode = 2: trains both on original and augmented images
def train(model, iter, tr, split_proportion, batch_size, grid_size, augment_times, mask_size, result_checker=None, graph_loss=False):
  for i in range(iter):
    tr_tar, tr_src = load_data.aug_datasets(tr, split_proportion, batch_size, grid_size, augment_times, mask_size)
    tr_tar = load_data.get_dataloader(tr_tar, batch_size, augment=True, shuffle=True)
    tr_src = load_data.get_dataloader(tr_src, batch_size, shuffle=True)
    #we will pre-process(augment) the data in order to prevent overloading gpu
    tr_tar  = load_data.pre_augment(tr_tar, batch_size=config.BATCH_SIZE, shuffle=True)
    tr_tar = load_data.DataLoader(tr_tar, config.BATCH_SIZE, shuffle=True)
    run.run_model(model, tr_src, tr_tar, config.GRID_SIZE, result_checker = result_checker, graph_loss=graph_loss)
    