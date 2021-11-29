import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class model(nn.Module):
    def __init__(self, init_grid, grid_size,maxfeat):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv = None
        self.conv = nn.Sequential (
          nn.MaxPool3d (2),
          nn.Conv3d (2, 20, 2),
          nn.ReLU(),
          nn.MaxPool3d (2),
          nn.Conv3d (20, 20, 2),
          nn.ReLU(),
          )
        self.linear1 = nn.Sequential(nn.Linear(4320,800),nn.ReLU(),nn.Linear(800, 100), nn.ReLU())
        self.linear2 = nn.Linear(100, 3*grid_size*grid_size*grid_size)
        self.linear2.bias = nn.Parameter(init_grid)
        self.linear2.weight.data.fill_(float(0))
            
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        