import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from imgaug import augmenters as iaa
from . import ops_3d, model


#perform cumsum operation, then upsample the grid to vox_size
class cumsum_layer_3d(nn.Module):
  def __init__(self, grid_size, vox_size, checker_board = False):
    super().__init__()
    self.upsampler = nn.Upsample(size = [vox_size,vox_size,vox_size], mode = 'trilinear')
    self.grid_offset_x = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True) 
    self.grid_offset_y = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True)
    self.grid_offset_z = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True)
    self.grid_offset_x = nn.Parameter(self.grid_offset_x)
    self.grid_offset_y = nn.Parameter(self.grid_offset_y)
    self.grid_offset_z = nn.Parameter(self.grid_offset_z)
    self.grid_size = grid_size
    
  def forward(self, x):
    #perform the cumsum operation to restore the original grid from the differential grid
    x = ops_3d.cumsum_3d(x, self.grid_offset_x, self.grid_offset_y, self.grid_offset_z)
    #Upsample the grid_size x grid_size warp field to image_size x image_size warp field
    x = self.upsampler(x)
    #shape (N,C,D,H,W)
    return x
    
    
class warp_layer_3d(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x, src_batch):
    x = x.permute(0,2,3,4,1)
    #calculate target estimation
    #input of shape (N,C,D,H,W)
    #grid of shape (N,D,H,W,3)
    x = nn.functional.grid_sample(src_batch.unsqueeze(0).permute([1,0,2,3,4]), x, mode='bilinear')
    return x

#a layer that ensure axial monotinicity
class axial_layer_3d(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.grid_size = grid_size
  def forward(self, x):
    #enforce axial monotinicity using the abs operation
    x = torch.abs(x)
    batch, grid = x.shape
    x = x.view(batch, 3,self.grid_size,self.grid_size,self.grid_size)
    return x


#define the model class
class ALIGNet_3d(nn.Module):
  def __init__(self, name, grid_size, vox_size, init_grid, model_idx):
    super().__init__()
    self.name = name 
    self.conv_layer = model.model(init_grid, grid_size, model_idx)
    self.cumsum_layer = cumsum_layer_3d(grid_size, vox_size)
    self.warp_layer = warp_layer_3d()
    self.axial_layer = axial_layer_3d(grid_size)
  #returns a differential grid
  def forward(self, x):
    x = self.conv_layer(x)
    x = self.axial_layer(x)
    return x
  def cumsum(self, diff_grid):
    x = self.cumsum_layer(diff_grid)
    return x 
  def warp(self, def_grid, src_batch):
    x = self.warp_layer(def_grid, src_batch)
    return x

