import os
import sys
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import config
import torch 
from torch.autograd import Variable
import torch.optim as optim
import model
import load_data
import validate
from utils import grid_helper
from utils.loss_functions import L2_Loss, L_TV_Loss, get_loss, avg_loss
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
    

#run the model inmultiple gpus
#data_loader can either be a single data_loader or a iterable list of data loaders (if so, source loader should also be a list of data loaders)
def run_epoch(rank, model, optimizer,  source_loader, data_loader, grid_size, image_size):
  #raise error if input types dont match
  if not type(data_loader) == type(source_loader):
    print('ERROR: data_laoder and source_loader must be of the same dtype')
    return None
  #if single value is given, convert data and source loader to list
  elif not type(data_loader) is list:
    data_loader = [data_loader]
    source_loader = [source_loader]
  loss_list = []
  #iterate over the list
  for j in range(len(data_loader)):
    tar_loader = data_loader[j]
    src_loader = source_loader[j]
    #raise error if the length of tar and src loader dont match
    if not len(tar_loader) == len(src_loader):
      print('ERROR: The length of tar_loader and src_loader does not match')
      return
    #define the iterables  
    tar_iter = iter(tar_loader)
    src_iter = iter(src_loader)
    for k in range(len(tar_loader)-1):
      aug_batch, tar_batch = next(tar_iter)
      src_batch = next(src_iter)
      aug_batch = aug_batch.squeeze(dim=1)
      #warning: significant overhead
      tar_batch = tar_batch.squeeze(dim=1).to(rank)
      src_batch = src_batch.squeeze(dim=1)
      input_image = torch.stack([src_batch, aug_batch])
      input_image  = input_image.permute([1,0,2,3])
      #run the network
      tar_est, diff_grid = model.forward(input_image, src_batch)
      tar_est = tar_est.squeeze(dim=1)
      loss = get_loss(tar_batch, tar_est, grid_size, diff_grid, image_size, rank)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_list.append(loss)
    time_avg = sum(time_) / k
  return loss_list

def run_process(rank, world_size, model,source_set, target_set, epochs, batch_size, grid_size, image_size, shuffle=False, result_checker=None, graph_loss=False):
  setup(rank, world_size)
  model = model.to(rank)
  #wrap the model with DDP module 
  model = DDP(model, device_ids=[rank])
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  #get the distributed dataloaders 
  target_loader = load_data.get_dataloader_parallel(target_set, batch_size, shuffle=shuffle)
  source_loader = load_data.get_dataloader_parallel(source_set, batch_size, shuffle=shuffle)
  epoch_loss = []
  for i in range(epochs):
    target_loader.sampler.set_epoch(i)
    source_loader.sampler.set_epoch(i)
    loss_list = run_epoch(rank, model, optimizer, source_loader, target_loader, grid_size, image_size)
    #set a barrier before updating resulting checker => and hence before saving the model
    #dist.barrier()
    #update only on process 0
    if rank == 0:
      avg_epoch_loss = avg_loss(loss_list)
      print(f'Loss in Epoch {i}: {avg_epoch_loss}')
      if not result_checker == None:
        result_checker.update(avg_epoch_loss)
        print(f'Validation Loss in Epoch {i}: {result_checker.avg_loss}')
  #save only if process id is 0
  if rank == 0:
    if (not result_checker == None) and graph_loss:
      print("Printing graph..")
      result_checker.print_graph(save_path = './' + model.name + '/outputs/loss_graphs/')
  cleanup()


def run_parallel(world_size, model, source_loader, target_loader, epochs, batch_size, grid_size, image_size, shuffle=False, result_checker=None, graph_loss=False):
  torch.multiprocessing.spawn(
    run_process,
    (world_size, model, source_loader, target_loader, epochs, batch_size, grid_size, image_size, shuffle, result_checker, graph_loss), 
    nprocs=world_size,
    join=True,
    )
    
    
