import torch 
import config
import grid_helper


def L2_Loss(target_image, warped_image):
  sum_check = torch.norm(target_image-warped_image, p=2)/config.BATCH_SIZE
  sum_check = sum_check/(128*128)
  L2_Loss = sum_check
  return L2_Loss

def L_TV_Loss(diff_grid, grid_size, lambda_):
  #create the identity differential grid  
  batch, _,w,h = diff_grid.shape
  diff_i_grid = grid_helper.init_grid(grid_size)
  diff_i_grid = diff_i_grid.view(2,grid_size,grid_size)
  diff_i_grid_x = diff_i_grid[0]
  diff_i_grid_y = diff_i_grid[1]
  L_TV_Loss = torch.norm(diff_grid[:,0] - diff_i_grid_x, p=1)/batch + torch.norm(diff_grid[:,1] - diff_i_grid_y, p=1)/batch
  L_TV_Loss = L_TV_Loss / (w*h) 
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss