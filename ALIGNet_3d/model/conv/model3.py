import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, init_grid, grid_size, maxfeat):
        super().__init__()
        self.init_grid = init_grid
        self.grid_size = grid_size
        self.maxfeat = maxfeat
        self.flatten = nn.Flatten()
        self.conv = None
        self.conv = nn.Sequential (
            nn.Conv3d(2, maxfeat, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(2),
            nn.Conv3d(maxfeat,maxfeat, 3, padding=1),
            nn.ReLU(True),
            nn.Conv3d(maxfeat,maxfeat, 2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(maxfeat,maxfeat, 2, padding=1),
            nn.ReLU(True),
            )
        self.linear1 = nn.Sequential(
          nn.Linear(maxfeat*grid_size*grid_size*grid_size, maxfeat),
          nn.ReLU(True)
          )
        self.linear2 = nn.Linear(maxfeat, 3*grid_size*grid_size*grid_size)
        self.linear2.bias = nn.Parameter(init_grid)
        self.linear2.weight.data.fill_(float(0))
            
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        