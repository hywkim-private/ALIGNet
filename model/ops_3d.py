#CORE HELPER FUNCTIONS
import torch
import numpy as np


#initialize the differential grid
#the parameter learn offset will define whether or not to learn the offset values during training
def init_grid_3d(grid_size):
  #spacing of the grid
  #-1 is because we have a1 = -1 (and thus there are grid_size - 1 "spacing" grids)
  delta = 2/(grid_size-1)
  np_grid = np.arange(grid_size, dtype=float)
  np_grid = np.full_like(np_grid,float(delta))
  ts_grid_x = torch.FloatTensor(np_grid)
  ts_grid_y = torch.FloatTensor(np_grid)
  ts_grid_z = torch.FloatTensor(np_grid)

  diff_i_grid_x, diff_i_grid_y, diff_i_grid_z = torch.meshgrid(ts_grid_x,ts_grid_y,ts_grid_z)
  diff_grid = torch.stack([diff_i_grid_x, diff_i_grid_y, diff_i_grid_z])
  diff_grid = diff_grid.view(3*grid_size*grid_size*grid_size)
  return diff_grid

#TODO: CHECK IF THE DIMENSIONS HERE ARE RIGHT

#perform cumsum operation on a 3d batch of inputs
#takes in grid tensors of shape batch x 3 x grid x grid x grid
#return grid tensors of shape batch x 3 x grid x grid x grid  
def cumsum_3d(grid, grid_offset_x, grid_offset_y, grid_offset_z):
  batch_size, dim, grid_1, grid_2, grid_3 = grid.shape
  #we start with dimension 1 in order to skip the batch dimension
  Integrated_grid_x = torch.cumsum(grid[:,0], dim = 3) + grid_offset_x
  Integrated_grid_y = torch.cumsum(grid[:,1], dim = 2) + grid_offset_y
  Integrated_grid_z = torch.cumsum(grid[:,2], dim = 1) + grid_offset_z

  Integrated_grid = torch.stack([Integrated_grid_x, Integrated_grid_y, Integrated_grid_z])
  Integrated_grid = Integrated_grid.permute([1,0,2,3,4])
  return Integrated_grid
    