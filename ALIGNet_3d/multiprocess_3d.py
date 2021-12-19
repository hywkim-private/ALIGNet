import os
import sys
import torch 
from torch.autograd import Variable
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import config_3d
from model import loss_3d, io_3d, ops_3d
from preprocess import datasets
import time
  
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    # initialize the process group
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:12359',
                            world_size=world_size, 
                            rank=rank)
  

def cleanup():
    dist.destroy_process_group()
      
  
#run the model for a single epoch
#data_loader can either be a single data_loader or a iterable list of data loaders (in such case, source loader should also be a list of data loaders)
def run_epoch_3d(rank, model, optimizer,  src_loader, tar_loader, grid_size, vox_size, lambda_):
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
      aug_batch = aug_batch.squeeze(dim=1)
      tar_batch = tar_batch.squeeze(dim=1).to(rank)
      src_batch = src_batch.squeeze(dim=1)
      input_image = torch.stack([src_batch, aug_batch])
      #input should be of shape (N,C,D,)
      input_image  = input_image.permute([1,0,2,3,4])
      #run the network
      diff_grid, def_grid, tar_est = model.forward(input_image, src_batch)
      tar_est = tar_est.squeeze(dim=1)
      init_grid = ops_3d.init_grid_3d(grid_size).to(rank)
      loss = loss_3d.get_loss_3d(tar_batch, tar_est, diff_grid, init_grid, lambda_, grid_size, vox_size)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_list.append(loss)
  return loss_list

def run_process(rank, world_size, model,  src_set, tar_set, grid_size, vox_size, batch_size, epochs, step, lambda_, result_checker=None, graph_loss=False):
  setup(rank, world_size)
  model = model.to(rank)
  model = DDP(model, device_ids=[rank])
  optimizer = optim.Adam(model.parameters(), lr=step)
  #get parallel dataloaders
  tar_loader = datasets.get_dataloader_parallel(tar_set, batch_size, shuffle=True)
  src_loader = datasets.get_dataloader_parallel(src_set, batch_size, shuffle=True)
  epoch_loss = []
  for i in range(epochs):
    start = time.time()
    loss_list = run_epoch_3d(rank, model, optimizer, src_loader, tar_loader, grid_size, vox_size, lambda_)
    if rank == 0:
      avg_epoch_loss = loss_3d.avg_loss_3d(loss_list)
      print(f'Loss in Epoch {i}: {avg_epoch_loss}')
      if not result_checker == None:
        result_checker.update(avg_epoch_loss)
        print(f'Validation Loss in Epoch {i}: {result_checker.avg_loss}')
    end = time.time()
    if rank == 0:
      print(end-start)
    """if (not result_checker == None) and graph_loss:
    print("Printing graph..")
    result_checker.print_graph(save_path = './' + model.name + '/outputs/loss_graphs/')"""
  
def run_parallel(world_size, model, src_set, tar_set, grid_size, vox_size, batch_size, epochs, step, lambda_, result_checker=None, graph_loss=False):
  torch.multiprocessing.spawn(
    run_process,
    (world_size, model, src_set, tar_set, grid_size, vox_size, batch_size, epochs, step, lambda_, result_checker, graph_loss), 
    nprocs=world_size,
    join=True,
    )
    
