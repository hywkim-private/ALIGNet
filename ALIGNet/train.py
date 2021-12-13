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
def train(model, iter, tr, val, test, result_checker=None, graph_loss=False):
  for i in range(iter):
      tr_tar_ds_aug, tr_src_ds_aug, val_tar_ds_aug, val_src_ds_aug, test_tar_ds_aug, test_src_ds_aug = load_data.aug_datasets(tr, val, test, augment=True)
      tr_tar_dl_aug = load_data.get_dataloader(tr_tar_ds_aug, augment=True, shuffle=True)
      tr_src_dl_aug = load_data.get_dataloader(tr_src_ds_aug, shuffle=True)
      #we will pre-process(augment) the data in order to prevent overloading gpu
      tr_tar_ds_aug  = load_data.pre_augment(tr_tar_dl_aug, batch_size=config.BATCH_SIZE, shuffle=True)
      tr_tar_dl_aug = load_data.DataLoader(tr_tar_ds_aug, config.BATCH_SIZE, shuffle=True)
      tr_tar = tr_tar_dl_aug
      tr_src = tr_src_dl_aug
    run.run_model(model, tr_src, tr_tar, config.GRID_SIZE, result_checker = result_checker, graph_loss=graph_loss)
    