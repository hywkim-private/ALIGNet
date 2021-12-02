#Loss functions
import torch 
from . import ops_3d

def L2_Loss_3d(target_vox, warped_vox, vox_size):
  batch,_,_,_ = target_vox.shape
  L2_Loss  = 0
  #sum loss values over the batch dimensions 
  for i in range(batch):
    tar_flat = torch.flatten(target_vox[i])
    est_flat = torch.flatten(warped_vox[i])
    loss  = torch.linalg.norm(torch.abs(tar_flat - est_flat)/(vox_size**3))
    L2_Loss += loss
  sum_check = L2_Loss/batch
  L2_Loss = sum_check
  return L2_Loss

def L_TV_Loss_3d2(diff_grid, grid_size, init_grid, lambda_):
  #create the identity differential grid  
  batch, _,d,h,w = diff_grid.shape
  diff_flat = torch.flatten(diff_grid)
  L_TV_Loss  = 0
  #sum loss values over the batch dimensions 
  for i in range(batch):
    diff_flat = torch.flatten(diff_grid[i])
    loss  = torch.linalg.norm(torch.abs(diff_flat - init_grid), float('inf'))
    L_TV_Loss += loss
  L_TV_Loss = L_TV_Loss / batch
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss
  

def L_TV_Loss_3d3(diff_grid, grid_size, init_grid, lambda_):
  #create the identity differential grid  
  batch, _,d,h,w = diff_grid.shape
  init_grid = init_grid.view(3,d,h,w)
  init_grid = init_grid[0].flatten()
  diff_flat = torch.flatten(diff_grid)
  L_TV_Loss  = 0
  #sum loss values over the batch dimensions 
  for i in range(batch):
    diff_flat_x = torch.flatten(diff_grid[i,0])
    diff_flat_y = torch.flatten(diff_grid[i,1])
    diff_flat_z = torch.flatten(diff_grid[i,2])
    loss_x  = torch.linalg.norm(torch.abs(diff_flat_x - init_grid), float('inf'))
    loss_y  = torch.linalg.norm(torch.abs(diff_flat_y - init_grid), float('inf'))
    loss_z  = torch.linalg.norm(torch.abs(diff_flat_z - init_grid), float('inf'))
    loss = loss_x + loss_y + loss_z
    L_TV_Loss += loss
  L_TV_Loss = L_TV_Loss / batch
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss
  
def L_TV_Loss_3d(diff_grid, grid_size, init_grid, lambda_):
  #create the identity differential grid  
  batch, _,d,h,w = diff_grid.shape
  L_TV_Loss  = 0
  #sum loss values over the batch dimensions 
  for i in range(batch):
    diff_flat = torch.flatten(diff_grid[i])
    loss  = torch.sum(torch.abs(diff_flat - init_grid))
    L_TV_Loss += loss
  L_TV_Loss = L_TV_Loss / (d*h*w)
  L_TV_Loss = L_TV_Loss / batch
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss
  
def get_loss_3d(tar_img, tar_est, diff_grid, init_grid, tv_lambda, grid_size, vox_size):
  L2_Loss_ = L2_Loss_3d(tar_img, tar_est, vox_size)
  L_TV_Loss_ = L_TV_Loss_3d(diff_grid, grid_size, init_grid, tv_lambda)
  loss = L_TV_Loss_ + L2_Loss_
  return loss
    
#get an average loss of all the elements in the loss_list 
def avg_loss_3d(loss_list):
  avg_loss = sum(loss_list)/len(loss_list)
  avg_loss = avg_loss.item()
  return avg_loss
  