#CORE HELPER FUNCTIONS
import torch
import numpy as np
import math
from scipy.interpolate import interpn, griddata,RegularGridInterpolator
from scipy.ndimage import affine_transform

from torch.nn.functional import grid_sample
#initialize the differential grid
#the parameter learn offset will define whether or not to learn the offset values during training
#model parameter specifies whether the grid is being used for the initialization of model parameters
def init_grid_3d(grid_size, model=False):
  #spacing of the grid
  #-1 is because we have a1 = -1 (and thus there are grid_size - 1 "spacing" grids)
  delta = 2/(grid_size-1)
  np_grid = np.zeros(grid_size, dtype=float)
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
  x_interp = RegularGridInterpolator(points, grids[0])
  y_interp = RegularGridInterpolator(points, grids[1])
  z_interp = RegularGridInterpolator(points, grids[2])
  z_verts = z_interp(meshes)
  y_verts = y_interp(meshes)
  x_verts = x_interp(meshes)
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
  min_p = reg_grid[min_i]
  max_p = reg_grid[max_i]
  reg_space = reg_grid[1]-reg_grid[0]
  while max_p-min_p == 0:
    #print(f"min_p and max_p: {min_i}, {max_i}")
    #if max_i == len(warp_grid):
    return interp_1d(reg_grid, warp_grid, (min_i,max_i), target_point)
  interp_weight = (max_p - target_point)/(max_p - min_p)
  #print(f"min_p:{min_p}, max_p:{max_p}, interp_weight:{interp_weight}")
  #interpolated location of the target_Point
  interp_loc = reg_grid[max_i] - interp_weight * reg_space
  #print(f"interp_loc: {interp_loc}")
  return interp_loc

#the variation of interp_1d for values in the right edge
#in this case window indicates second rightmost and the rightmost value
def exterp_1d_ledge(reg_grid, warp_grid, window, target_point):
  min_i, max_i = window
  min_p = reg_grid[min_i]
  max_p = reg_grid[max_i]
  reg_space = reg_grid[1]-reg_grid[0]
  while math.isclose(min_p,max_p,abs_tol=1e-6):
    #print(f"min_p and max_p: {min_i}, {max_i}")
    max_i += 1
    return exterp_1d_ledge(reg_grid, warp_grid, (min_i,max_i), target_point)
  #this will be a negative value
  exterp_weight = (min_p - target_point)/(max_p - min_P)
  #interpolated location of the target_Point
  exterp_loc = reg_grid[min_i] + exterp_weight * reg_space
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
  #minimum value of the regular grid
  min_reg = x_reg[0]
  #maximum value of the regular grid
  max_reg = x_reg[max_idx]
  x_fr = np.zeros([len(X), len(X), len(X)])
  y_fr = np.zeros([len(X), len(X), len(X)])
  z_fr = np.zeros([len(X), len(X), len(X)])
  for i in range(len(X)):
    for j in range(len(Y)):
      x_fr[:,i,j] = griddata(X[:,i,j], x_reg[:,i,j],x_reg[:,i,j],fill_value="extrapolate")
      y_fr[i,:,j] = griddata(Y[i,:,j], y_reg[i,:,j],y_reg[i,:,j],fill_value="extrapolate")
      z_fr[i,j,:] = griddata(Z[i,j,:], z_reg[i,j,:],z_reg[i,j,:],fill_value="extrapolate")
  #todo: interpolation method
  #return the forward warp grids
  return x_fr, y_fr, z_fr
              
              