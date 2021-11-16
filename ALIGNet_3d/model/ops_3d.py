#CORE HELPER FUNCTIONS
import torch
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import affine_transform

from torch.nn.functional import grid_sample
#initialize the differential grid
#the parameter learn offset will define whether or not to learn the offset values during training
#model parameter specifies whether the grid is being used for the initialization of model parameters
def init_grid_3d(grid_size, model=False):
  #spacing of the grid
  #-1 is because we have a1 = -1 (and thus there are grid_size - 1 "spacing" grids)
  delta = 2/(grid_size-1)
  np_grid = np.arange(grid_size, dtype=float)
  np_grid = np.full_like(np_grid,float(delta))
  ts_grid_x = torch.FloatTensor(np_grid)
  ts_grid_y = torch.FloatTensor(np_grid)
  ts_grid_z = torch.FloatTensor(np_grid)
  
  diff_i_grid_x, diff_i_grid_y, diff_i_grid_z = torch.meshgrid(ts_grid_x,ts_grid_y,ts_grid_z)
  #must be of shape (D, H, W)
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
  Integrated_grid_x = torch.cumsum(grid[:,0], dim = 1) + grid_offset_x
  Integrated_grid_y = torch.cumsum(grid[:,1], dim = 2) + grid_offset_y
  Integrated_grid_z = torch.cumsum(grid[:,2], dim = 3) + grid_offset_z
  Integrated_grid = torch.stack([Integrated_grid_x, Integrated_grid_y, Integrated_grid_z])
  Integrated_grid = Integrated_grid.permute([1,0,2,3,4])
  return Integrated_grid
    


#input:
#mesh: mesh vertice of shape np.arrray(n_points, 3)
#grid: deformation grid of shape (3, n, n, n)
#output: deformed mesh of shape (n_points, 3)
#EXPLANATION: the function interpolate_3d_mesh, given grid index points and data value corressponding to these points,
#returns the expected data values for input coordinate points in the shape, [x, y, z].
#We, however, need to get the "deformed" coordinate points for each coordinate input [x,y,z]. Since our input "data values"
#represent the end coordinates of the deformed meshes along each axis (in shape [3,n,n,n]), we need 3 interpolate_3d_mesh 
#functions that each retrieves the "deformed" coordinates for x, y, and z  axis. 
#For example, the "deformed" coordinates for [0.4, 0.5, 0.1], we will be [x_interp([0.4, 0.5, 0.1], y_interp[0.4, 0.5, 0.1], x_interp[0.4, 0.5, 0.1])].
def interpolate_3d_mesh(meshes, grids, vox_size):
  #define the grid points of the 3d mesh
  x_points = np.linspace(-1, 1,32)
  y_points = np.linspace(-1, 1,32)
  z_points = np.linspace(-1, 1,32)
  points = (x_points, y_points, z_points)
  #define the interpolation function along x, y, and z axis
  x_verts = interpn(points, grids[0], meshes, fill_value=0)
  y_verts = interpn(points, grids[1], meshes, fill_value=0)
  z_verts = interpn(points, grids[2], meshes, fill_value=0)
  deformed_batch = []
  mesh_list = []
  for i in range(len(x_verts)):
    vert = np.array([z_verts[i], y_verts[i], x_verts[i]])
    mesh_list.append(vert)
  deformed_verts = np.stack(mesh_list, axis=0)
  return deformed_verts

#given a window of range (index), meshgrid, and a target point, get the interpolated value from the window 
#input
#window: tuple that defines the index window of the warp field
#grid: 1d grid of shape (x) that defines the warp field
#target_point: the value of the target point
#return
#the interpolated location of the target value
def interp_1d(grid, window, target_point):
  min_i, max_i = window
  min_p = grid[min_i]
  max_p = grid[max_i]
  #handle for the index 0 extrapolation case where index 0 and 1 are identical
  while min_p == max_p:
    max_p += 1
  interp_weight = (target_point - min_p)/(max_p - min_p)
  #interpolated location of the target_Point
  interp_loc = min_i + interp_weight
  return interp_loc



#the variation of interp_1d for values in the right edge
#in this case window indicates second rightmost and the rightmost value
def exterp_1d_redge(grid, window, target_point):
  min_i, max_i = window
  min_p = grid[min_i]
  max_p = grid[max_i]
  while min_p == max_p:
    min_p -= 1
  #this will be a negative value
  exterp_weight = (max_p - target_point)/(max_p - min_p)
  #interpolated location of the target_Point
  exterp_loc = max_i + exterp_weight
  return exterp_loc
  
#given the 3d backward warp mesh X,Y,Z
#get the forward warp field
#input
#X,Y,Z: 3d mesh grid along each axis, in the shape of (grid_size, grid_size, grid_size)
def convert_to_forward_warp(X,Y,Z):
  #logic: using the interpn functionality of scipy, interpolate the regular coordinate grids from the backward warpfields,
  #thereby converting the backwarp warp into forward
  max_idx = len(X) - 1
  #define the regular grid
  ls = np.linspace(-1, 1,32)
  x_reg, y_reg, z_reg = np.meshgrid(ls,ls,ls)
  x_fr = np.zeros([len(X), len(X), len(X)])
  y_fr = np.zeros([len(X), len(X), len(X)])
  z_fr = np.zeros([len(X), len(X), len(X)])
  #the spacing of the regular grid
  reg_space = ls[1] - ls[2]
  for i in range(len(X)):
    for j in range(len(X)):
      for k in range(len(X)):
        #if the diff value is positive and smaller than regular linspace, we define it as being placed at the correct range
        #x_win_set defines whether or not the window range for x has been found
        idx_x = k 
        x_win_set = False
        #specifies whether the given idx is an edge
        x_edge_r = False
        while x_win_set == False:
        
          if idx_x == max_idx:
              x_edge_r = True 
              x_win_set = True
          elif x_reg[k,i,j] >= X[idx_x,i,j] and x_reg[k,i,j] < X[idx_x+1,i,j]:
            #define the window of range wherein x value lies
            x_win = (idx_x, idx_x+1)
            #set x_win_set to True in order to move on
            x_win_set = True 
          else:
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
            idx_x += 1
     
        #need a different  function for the right edge case
        if x_edge_r:
          x_fr[k,i,j] = exterp_1d_redge(X[:,i,j],(idx_x-1, idx_x), x_reg[k,i,j])
        elif x_win_set:
          x_fr[k,i,j] = interp_1d(X[:,i,j], x_win, x_reg[k,i,j])
        #if the diff value is positive and smaller than regular linspace, we define it as being placed at the correct range
        #y_win_set defines whether or not the window range for x has been found

        idx_y = k 
        y_win_set = False
        #specifies whether the given idx is an edge
        y_edge_l = False
        y_edge_r = False
        while y_win_set == False:
          if idx_y == max_idx:
              y_edge_r = True 
              y_win_set = True
          elif y_reg[i,k,j] >= Y[i,idx_y, j] and y_reg[i,k,j] < Y[i,idx_y+1,j]:
            #define the window of range wherein x value lies
            y_win = (idx_y, idx_y+1)
            #set x_win_set to True in order to move on
            y_win_set = True 
          else:
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
            idx_y += 1
        #need a different  function for the right edge case
        if y_edge_r:
          y_fr[i,k,j] = exterp_1d_redge(Y[i,:,j],(idx_y-1, idx_y), y_reg[i,k,j])
        elif y_win_set:
          y_fr[i,k,j] = interp_1d(Y[i,:,j], y_win, y_reg[i,k,j])

        #if the diff value is positive and smaller than regular linspace, we define it as being placed at the correct range
        #x_win_set defines whether or not the window range for x has been found
        idx_z = k 
        z_win_set = False
        #specifies whether the given idx is an edge
        z_edge_r = False
        while z_win_set == False:
          if idx_z == max_idx:
              z_edge_r = True 
              z_win_set = True
          elif z_reg[i,j,k] >= Z[i,j,idx_z] and z_reg[i,j,k] < Z[i,j,idx_z+1]:
            #define the window of range wherein x value lies
            z_win = (idx_z,idx_z+1)
            #set x_win_set to True in order to move on
            z_win_set = True 
          else:
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
            idx_z += 1
            #case for when x diff is bigger than reg space 

        #need a different  function for the right edge case
        if z_edge_r:
          z_fr[i,j,k] = exterp_1d_redge(Z[i,j,:],(idx_z-1, idx_z), z_reg[i,j,k])
        elif x_win_set:
          z_fr[i,j,k] = interp_1d(Z[i,j,:], z_win, z_reg[i,j,k])
  #return the forward warp grids
  return x_fr, y_fr, z_fr
              
              