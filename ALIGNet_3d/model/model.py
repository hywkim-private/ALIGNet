import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#define the alignnet model
def get_conv_3d(conv_index):
    model1 = nn.Sequential (
      nn.MaxPool3d (2),
      nn.Conv3d (2, 20, 2),
      nn.ReLU(),
      nn.MaxPool3d (2),
      nn.Conv3d (20, 20, 2),
      nn.ReLU(),
      )
  
    model2 = nn.Sequential (
        nn.MaxPool3d(2),
        nn.Conv3d(2, 20, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(20,40,3,padding=1),
        nn.ReLU(),
        nn.MaxPool3d(2),
        nn.Conv3d(40,60,3,padding=1),
        nn.ReLU(),
        )
    if conv_index == 1:
        model = model1
    elif conv_index == 2:
        model = model2
    return model

#define the linear layer
def get_linear1_3d(linear1_idx, grid_size, maxfeat):
    linear1_1 = nn.Sequential(nn.Linear(4320,800),nn.ReLU(),nn.Linear(800, 100), nn.ReLU())
    linear1_2 = nn.Sequential(
      nn.Linear(30720, maxfeat),
      nn.ReLU()
      )
    if linear1_idx == 1:
        linear1 = linear1_1
    elif linear1_idx == 2:
        linear1 = linear1_2 
    return linear1

def get_linear2_3d(linear2_idx, init_grid, grid_size, maxfeat):
    linear2_1 = nn.Linear(100, 3*grid_size*grid_size*grid_size)
    linear2_2 = nn.Linear(maxfeat, 3*grid_size*grid_size*grid_size)
    if linear2_idx == 1:
        linear2 = linear2_1
    elif linear2_idx == 2:
        linear2 = linear2_2
    linear2.bias = nn.Parameter(init_grid)
    linear2.weight.data.fill_(float(0))
    return linear2

class model(nn.Module):
    def __init__(self, init_grid, grid_size, model_idx):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv = None
        if model_idx == 1:
            self.conv = get_conv_3d(1)
            self.linear1 = get_linear1_3d(1, grid_size,0)
            self.linear2 = get_linear2_3d(1, init_grid, grid_size,0)
        elif model_idx == 2:
            self.conv = get_conv_3d(2)
            self.linear1 = get_linear1_3d(2, grid_size, 60)
            self.linear2 = get_linear2_3d(2, init_grid,grid_size, 60)
            
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        