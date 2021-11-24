#Loss functions
import torch 
from . import ops_3d

def L2_Loss_3d(target_vox, warped_vox, vox_size):
  batch,_,_,_ = target_vox.shape
  sum_check = torch.norm(target_vox - warped_vox, p=2)/batch
  sum_check = sum_check/(vox_size**3)
  L2_Loss = sum_check
  return L2_Loss

def L_TV_Loss_3d(diff_grid, grid_size, init_grid, lambda_):
  #create the identity differential grid  
  batch, _,d,h,w = diff_grid.shape
  diff_i_grid = init_grid.view(3,grid_size,grid_size,grid_size)
  diff_i_grid_x = diff_i_grid[0]
  diff_i_grid_y = diff_i_grid[1]
  diff_i_grid_z = diff_i_grid[2]
  L_TV_Loss = torch.norm(diff_grid[:,0] - diff_i_grid_x, p=1) + torch.norm(diff_grid[:,1] - diff_i_grid_y, p=1) + torch.norm(diff_grid[:,2] - diff_i_grid_z, p=1)
  L_TV_Loss = L_TV_Loss / batch
  L_TV_Loss = L_TV_Loss / (d*w*h) 
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss
  
def get_loss_3d(tar_img, tar_est, diff_grid, init_grid, grid_size, vox_size):
  L2_Loss_ = L2_Loss_3d(tar_img, tar_est, vox_size)
  L_TV_Loss_ = L_TV_Loss_3d(diff_grid, grid_size, init_grid, 1e-3/5)
  loss = L_TV_Loss_ + L2_Loss_
  return loss
    
#get an average loss of all the elements in the loss_list 
def avg_loss_3d(loss_list):
  avg_loss = sum(loss_list)/len(loss_list)
  avg_loss = avg_loss.item()
  return avg_loss
  