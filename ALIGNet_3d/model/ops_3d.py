#CORE HELPER FUNCTIONS
import torch
import numpy as np
import math
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
  meshes = np.flip(meshes,1)
  #define the interpolation function along x, y, and z axis
  z_verts = interpn(points, grids[0], meshes, fill_value=0)
  y_verts = interpn(points, grids[1], meshes, fill_value=0)
  x_verts = interpn(points, grids[2], meshes, fill_value=0)
  deformed_batch = []
  mesh_list = []
  for i in range(len(x_verts)):
    vert = np.array([x_verts[i], y_verts[i], z_verts[i]])
    mesh_list.append(vert)
  deformed_verts = np.stack(mesh_list, axis=0)
  return deformed_verts

#given a window of range (index), meshgrid, and a target point, get the interpolated value from the window 
#input
#window: tuple that defines the index window of the warp field
#warp_grid: 1d grid of shape (x) that defines the warp field
#reg_grid: the regular (unwarped) grid 
#target_point: the value of the target point
#return
#the interpolated location of the target value
def interp_1d( reg_grid, warp_grid, window, target_point):
  min_i, max_i = window
  min_p = warp_grid[min_i]
  max_p = warp_grid[max_i]
  reg_space = reg_grid[1]-reg_grid[0]
  while max_p-min_p == 0:
    #print(f"min_p and max_p: {min_i}, {max_i}")
    #if max_i == len(warp_grid):
    return interp_1d(reg_grid, warp_grid, (min_i,max_i), target_point)
  interp_weight = (target_point - min_p)/(max_p - min_p)
  #print(f"min_p:{min_p}, max_p:{max_p}, interp_weight:{interp_weight}")
  #interpolated location of the target_Point
  interp_loc = reg_grid[min_i] + interp_weight * reg_space
  #print(f"interp_loc: {interp_loc}")
  return interp_loc

#the variation of interp_1d for values in the right edge
#in this case window indicates second rightmost and the rightmost value
def exterp_1d_ledge(reg_grid, warp_grid, window, target_point):
  min_i, max_i = window
  min_p = warp_grid[min_i]
  max_p = warp_grid[max_i]
  reg_space = reg_grid[1]-reg_grid[0]
  while math.isclose(min_p,max_p,abs_tol=1e-6):
    #print(f"min_p and max_p: {min_i}, {max_i}")
    max_i += 1
    return exterp_1d_ledge(reg_grid, warp_grid, (min_i,max_i), target_point)
  #this will be a negative value
  exterp_weight = (min_p - target_point)/(max_p - target_point)
  #interpolated location of the target_Point
  exterp_loc = reg_grid[min_i] - exterp_weight * reg_space
  return exterp_loc

#the variation of interp_1d for values in the right edge
#in this case window indicates second rightmost and the rightmost value
def exterp_1d_redge(reg_grid, warp_grid, window, target_point):
  min_i, max_i = window
  min_p = warp_grid[min_i]
  max_p = warp_grid[max_i]
  reg_space = reg_grid[1]-reg_grid[0]

  while math.isclose(min_p,max_p,abs_tol=1e-6):
    min_i -= 1
    return exterp_1d_redge(reg_grid,warp_grid, (min_i,max_i), target_point)
  #this will be a negative value
  exterp_weight = (target_point - max_p)/(target_point - min_p)
  
  #interpolated location of the target_Point
  exterp_loc = reg_grid[max_i] - exterp_weight * reg_space
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
  y_reg, x_reg,z_reg = np.meshgrid(ls,ls,ls)
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
        x_edge_l = False
        while x_win_set == False:
          if x_reg[k,i,j] > X[max_idx,i,j]:
            x_win = (max_idx-1, max_idx)
            x_edge_r = True 
            x_win_set = True 
          elif x_reg[k,i,j] < X[0,i,j]:
            x_win = (0, 1)
            x_edge_l = True 
            x_win_set = True
          elif x_reg[k,i,j] >= X[idx_x,i,j] and x_reg[k,i,j] <= X[idx_x+1,i,j]:
            #define the window of range wherein x value lies
            x_win = (idx_x, idx_x+1)
            #set x_win_set to True in order to move on
            x_win_set = True
          elif x_reg[k,i,j] < X[idx_x,i,j]:
            idx_x -= 1
          else:
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
            idx_x += 1
     
        #need a different  function for the right edge case
        if x_edge_r:
          x_fr[k,i,j] = exterp_1d_redge(x_reg[:,i,j],X[:,i,j],(idx_x-1, idx_x), x_reg[k,i,j])
        elif x_edge_l:
          x_fr[k,i,j] = exterp_1d_ledge(x_reg[:,i,j],X[:,i,j],x_win, x_reg[k,i,j])
        elif x_win_set:
          x_fr[k,i,j] = interp_1d(x_reg[:,i,j],X[:,i,j], x_win, x_reg[k,i,j])
        #if the diff value is positive and smaller than regular linspace, we define it as being placed at the correct range
        #y_win_set defines whether or not the window range for x has been found

        idx_y = k 
        y_win_set = False
        #specifies whether the given idx is an edge
        y_edge_r = False
        y_edge_l = False 
        while y_win_set == False:
          if y_reg[i,k,j] > Y[i,max_idx, j]:
            y_win = (max_idx-1, max_idx)
            y_edge_r = True 
            y_win_set = True 
          elif y_reg[i,k,j] < Y[i,0, j]:
            y_win = (0, 1)
            y_win_set = True
            y_edge_l = True
          elif y_reg[i,k,j] >= Y[i,idx_y, j] and y_reg[i,k,j] <= Y[i,idx_y+1,j]:
            #define the window of range wherein x value lies
            y_win = (idx_y, idx_y+1)
            #set x_win_set to True in order to move on
            y_win_set = True 
          elif y_reg[i,k,j] < Y[i,idx_y,j]:
              idx_y -= 1
          else:
            idx_y += 1
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
        #need a different  function for the right edge case
        if y_edge_r:
          y_fr[i,k,j] = exterp_1d_redge(y_reg[i,:,j],Y[i,:,j],(idx_y-1, idx_y), y_reg[i,k,j])
        elif y_edge_l:
          y_fr[i,k,j] = exterp_1d_ledge(y_reg[i,:,j], Y[i,:,j], y_win, y_reg[i,k,j])
        elif y_win_set:
          y_fr[i,k,j] = interp_1d(y_reg[i,:,j], Y[i,:,j], y_win, y_reg[i,k,j])

        #if the diff value is positive and smaller than regular linspace, we define it as being placed at the correct range
        #x_win_set defines whether or not the window range for x has been found
        idx_z = k 
        z_win_set = False
        #specifies whether the given idx is an edge
        z_edge_r = False
        z_edge_l = False
        while z_win_set == False:
          if z_reg[i,j,k] > Z[i, j, max_idx]:
            z_win = (max_idx-1, max_idx)
            z_edge_r = True 
            z_win_set = True 
          elif z_reg[i,j,k] < Z[i,j,0]:
            z_win = (0, 1)
            z_win_set = True 
            z_edge_l = True 
          elif z_reg[i,j,k] >= Z[i,j,idx_z] and z_reg[i,j,k] <= Z[i,j,idx_z+1]:
            #define the window of range wherein x value lies
            z_win = (idx_z,idx_z+1)
            #set x_win_set to True in order to move on
            z_win_set = True 
          elif z_reg[i,j,k] < Z[i,j,idx_z]:
            idx_z -= 1
          else:
            #if not, move the window to find the appropriate range
            #case for when diff is negative (x reg value is on smaller than the min range of window)
            idx_z += 1
            #case for when x diff is bigger than reg space 

        #need a different  function for the right edge case
        if z_edge_r:
          z_fr[i,j,k] = exterp_1d_redge(z_reg[i,j,:],Z[i,j,:],(idx_z-1, idx_z), z_reg[i,j,k])
        elif z_edge_l:
          z_fr[i,j,k] = exterp_1d_ledge(z_reg[i,j,:],Z[i,j,:], z_win, z_reg[i,j,k])
        elif z_win_set:
          z_fr[i,j,k] = interp_1d(z_reg[i,j,:],Z[i,j,:], z_win, z_reg[i,j,k])
  #return the forward warp grids
  return x_fr, y_fr, z_fr
              
              