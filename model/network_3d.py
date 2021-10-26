import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from imgaug import augmenters as iaa
from . import ops_3d

#define the alignnet model
def get_conv_3d(grid_size):
  model = nn.Sequential (
      nn.MaxPool3d (2),
      nn.Conv3d (2, 20, 2),
      nn.ReLU(),
      nn.MaxPool3d (2),
      nn.Conv3d (20, 20, 2),
      nn.ReLU(),
  )
  return model


class warp_layer_3d(nn.Module):
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


  def forward(self, x, src_batch):
    #perform the cumsum operation to restore the original grid from the differential grid
    x = ops_3d.cumsum_3d(x, self.grid_offset_x, self.grid_offset_y, self.grid_offset_z)
    #Upsample the grid_size x grid_size warp field to image_size x image_size warp field
    x = self.upsampler(x)
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

#define the convolutional + linear layers
class conv_layer_3d(nn.Module):
  def __init__(self, grid_size, init_grid):
    super().__init__()
    self.conv = get_conv_3d(grid_size)
    self.flatten = nn.Flatten()
    self.linear1 = nn.Sequential(nn.Linear(4320,800),nn.ReLU(),nn.Linear(800, 100), nn.ReLU())
    self.linear2 = nn.Linear(100, 3*grid_size*grid_size*grid_size)
    self.linear2.bias = nn.Parameter(init_grid)
    self.linear2.weight.data.fill_(float(0))
  def forward(self, x):
    #print(f"conv_layer_3d: shape of input-{x.shape}")
    x = self.conv(x)
    #print(f"conv_layer_3d: shape after the conv layer-{x.shape}")
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.linear2(x)
    
    return x

#define the model class
class ALIGNet_3d(nn.Module):
  def __init__(self, name, grid_size, vox_size,init_grid):
    super().__init__()
    self.name = name 
    self.conv_layer = conv_layer_3d(grid_size, init_grid)
    self.warp_layer = warp_layer_3d(grid_size, vox_size)
    self.axial_layer = axial_layer_3d(grid_size)
  #returns a differential grid
  def forward(self, x):
    x = self.conv_layer(x)
    x = self.axial_layer(x)
    return x
  def warp(self, diff_grid, src_batch):
    x = self.warp_layer(diff_grid, src_batch)
    return x

