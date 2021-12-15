import torch 
import config
from . import grid_helper


def L2_Loss(target_image, warped_image, image_size):
  batch,_,_ = target_image.shape
  sum_check = torch.norm(target_image - warped_image, p=2)/batch
  sum_check = sum_check/(image_size*image_size)
  L2_Loss = sum_check
  return L2_Loss

def L_TV_Loss(diff_grid, grid_size, lambda_, device):
  #create the identity differential grid  
  batch, _,w,h = diff_grid.shape
  diff_i_grid = grid_helper.init_grid(grid_size).to(device)
  diff_i_grid = diff_i_grid.view(2,grid_size,grid_size)
  diff_i_grid_x = diff_i_grid[0]
  diff_i_grid_y = diff_i_grid[1]
  L_TV_Loss = torch.norm(diff_grid[:,0] - diff_i_grid_x, p=1) + torch.norm(diff_grid[:,1] - diff_i_grid_y, p=1)
  L_TV_Loss = L_TV_Loss / batch
  L_TV_Loss = L_TV_Loss / (w*h) 
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss
  
def get_loss(tar_img, tar_est, grid_size, diff_grid, image_size, device):
  L2_Loss_ = L2_Loss(tar_img, tar_est, image_size)
  L_TV_Loss_ = L_TV_Loss(diff_grid, grid_size, 1e-3, device)
  loss = L_TV_Loss_ + L2_Loss_
  return loss
    
#get an average loss of all the elements in the loss_list 
def avg_loss(loss_list):
  avg_loss = sum(loss_list)/len(loss_list)
  avg_loss = avg_loss.item()
  return avg_loss
  