#import config
import torch 
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import config_3d
from model import loss_3d, io_3d, ops_3d
from preprocess import datasets
  
#run the model for a single epoch
#data_loader can either be a single data_loader or a iterable list of data loaders (in such case, source loader should also be a list of data loaders)
def run_epoch_3d(model, optimizer,  src_loader, tar_loader, grid_size):
  #raise error if input types dont match
  if not type(tar_loader) == type(src_loader):
    print('ERROR: data_laoder and source_loader must be of the same dtype')
    return None
  #if single value is given, convert data and source loader to list
  elif not type(tar_loader) is list:
    #print("run_epoch_3d: not a list")
    tar_loader_l = [tar_loader]
    src_loader_l = [src_loader]
  else:
    tar_loader_l = tar_loader
    src_loader_l = src_loader
  loss_list = []
  #iterate over the list
  for j in range(len(tar_loader_l)):
    tar_loader = tar_loader_l[j]
    src_loader = src_loader_l[j]
    #raise error if the length of tar and src loader dont match
    if not len(tar_loader) == len(src_loader):
      print('ERROR: The length of tar_loader and src_loader does not match')
      return
    #define the iterables
    tar_iter = iter(tar_loader)
    src_iter = iter(src_loader)
    for k in range(len(tar_loader)):
      aug_batch, tar_batch = next(tar_iter)
      src_batch = next(src_iter)
      aug_batch = torch.tensor(aug_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config_3d.DEVICE)
      tar_batch = torch.tensor(tar_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config_3d.DEVICE)
      src_batch = torch.tensor(src_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config_3d.DEVICE)
      input_image = torch.stack([src_batch, aug_batch])
      #input should be of shape (N,C,D,)
      input_image  = input_image.permute([1,0,2,3,4])
      #run the network
      diff_grid, def_grid, tar_est = model.forward(input_image, src_batch)
      tar_est = tar_est.squeeze(dim=1)
      init_grid = ops_3d.init_grid_3d(grid_size).to(config_3d.DEVICE)
      loss = loss_3d.get_loss_3d(tar_batch, tar_est, diff_grid, init_grid, config_3d.GRID_SIZE,  config_3d.VOX_SIZE)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_list.append(loss)
  return loss_list

def run_model(model,src_loader, tar_loader, grid_size, result_checker=None, graph_loss=False):
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  epoch_loss = []
  for i in range(config_3d.EPOCHS):
    loss_list = run_epoch_3d(model, optimizer, src_loader, tar_loader, grid_size)
    avg_epoch_loss = loss_3d.avg_loss_3d(loss_list)
    print(f'Loss in Epoch {i}: {avg_epoch_loss}')
    if not result_checker == None:
      result_checker.update(avg_epoch_loss)
      print(f'Validation Loss in Epoch {i}: {result_checker.avg_loss}')
    """if (not result_checker == None) and graph_loss:
    print("Printing graph..")
    result_checker.print_graph(save_path = './' + model.name + '/outputs/loss_graphs/')"""
  



#train the data
#data_index 0 = vase, 1= plane
#iter is the number of iterations
def train_3d(model, model_path, iter_t, tr, model_name, result_checker=None, graph_loss=False):
  for i in range(iter_t):
    tr_tar, tr_src = datasets.aug_datasets_3d(tr, 0, config_3d.TARGET_PROPORTION, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, augment_times=config_3d.AUGMENT_TIMES_TR)
    tr_tar_dl =  DataLoader(tr_tar, batch_size=config_3d.BATCH_SIZE, collate_fn=tr_tar.collate_fn, shuffle=True)
    tr_src_dl = DataLoader(tr_src, batch_size=config_3d.BATCH_SIZE, shuffle=True)
    tr_tar = tr_tar_dl
    tr_src = tr_src_dl
    run_model(model, tr_src, tr_tar, config_3d.GRID_SIZE, result_checker = result_checker, graph_loss=graph_loss)
    io_3d.save_model(model, model_path, model_name+'.pt')
    