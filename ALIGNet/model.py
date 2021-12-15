import config
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import augment
from imgaug import augmenters as iaa
from utils import grid_helper



#define the alignnet model
def get_conv(grid_size):
  model = nn.Sequential (
      nn.MaxPool2d (2),
      nn.Conv2d (2, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 2),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 5),
      nn.ReLU(),
   
  )
  return model


class warp_layer(nn.Module):
  def __init__(self, grid_size, checker_board = False):
    super().__init__()
    self.upsampler = nn.Upsample(size = [config.IMAGE_SIZE, config.IMAGE_SIZE], mode = 'bilinear')
    self.grid_offset_x = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True) 
    self.grid_offset_y = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True)
    self.grid_offset_x = nn.Parameter(self.grid_offset_x)
    self.grid_offset_y = nn.Parameter(self.grid_offset_y)
    self.grid_size = grid_size

  def forward(self, x, src_batch, checker_board=False):
    #perform the cumsum operation to restore the original grid from the differential grid
    x =grid_helper.cumsum_2d(x, self.grid_offset_x, self.grid_offset_y)
    #Upsample the grid_size x grid_size warp field to image_size x image_size warp field
    x = self.upsampler(x)
    x = x.permute(0,2,3,1)
    if checker_board:
      source_image = augment.apply_checkerboard(src_batch, config.IMAGE_SIZE)
    #calculate target estimation
    x = nn.functional.grid_sample(src_batch.unsqueeze(0).permute([1,0,2,3]), x, mode='bilinear')
    return x

#a layer that ensure axial monotinicity
class axial_layer(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.grid_size = grid_size
  def forward(self, x):
    #enforce axial monotinicity using the abs operation
    x = torch.abs(x)
    batch, grid = x.shape
    x = x.view(batch, 2,self.grid_size,self.grid_size)
    return x

#define the convolutional + linear layers
class conv_layer(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.conv = get_conv(grid_size)
    self.flatten = nn.Flatten()
    self.linear1 = nn.Sequential(nn.Linear(80,20),nn.ReLU(),)
    self.linear2 = nn.Linear(20, 2*grid_size*grid_size)
    self.linear2.bias = nn.Parameter(grid_helper.init_grid(grid_size).view(-1))
    self.linear2.weight.data.fill_(float(0))
  def forward(self, x):
    x = self.conv(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x

#define the model class
class ALIGNet(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.conv_layer = conv_layer(grid_size)
    self.warp_layer = warp_layer(grid_size)
    self.axial_layer = axial_layer(grid_size)\
  #returns a differential grid
  def forward(self, x):
    x = self.conv_layer(x)
    x = self.axial_layer(x)
    return x
    
  def warp(self, diff_grid, src_batch):
    x = self.warp_layer(diff_grid, src_batch)
    return x

