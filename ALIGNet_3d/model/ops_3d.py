#CORE HELPER FUNCTIONS
import torch
import numpy as np
import math
from scipy.interpolate import interpn, griddata,RegularGridInterpolator,LinearNDInterpolator
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
  ls = np.linspace(-1, 1, vox_size)
  y_reg, x_reg,z_reg = np.meshgrid(ls,ls,ls)
  #make grids into sets of data points
  warp_coords = np.stack((grids[2].flatten(), grids[1].flatten(), grids[0].flatten()), axis=1)
  #make regular grids into sets of data points
  reg_coords_x = x_reg.flatten()
  reg_coords_y = y_reg.flatten()
  reg_coords_z = z_reg.flatten()
  interp_x = LinearNDInterpolator(warp_coords, reg_coords_x)
  interp_y = LinearNDInterpolator(warp_coords, reg_coords_y)
  interp_z = LinearNDInterpolator(warp_coords, reg_coords_z)
  #define the interpolation function along x, y, and z axis
    #interpolate along xy plane
  x_verts = interp_x(meshes)
  y_verts = interp_y(meshes)
  z_verts = interp_z(meshes)
 
  deformed_batch = []
  mesh_list = []
  for i in range(len(x_verts)):
    vert = np.array([z_verts[i], y_verts[i], x_verts[i]])
    mesh_list.append(vert)
  deformed_verts = np.stack(mesh_list, axis=0)
  return deformed_verts

  
#FOR VISUALIZATION PURPOSES ONLY 
#given the 3d backward warp mesh X,Y,Z
#get the forward warp field
#input
#X,Y,Z: 3d mesh grid along each axis, in the shape of (grid_size, grid_size, grid_size)
def convert_to_forward_warp(grid):
  #get how many voxels are there in one axis
  vox_size = len(grid[0])
  #define the grid points of the 3d mesh
  ls = np.linspace(-1, 1, vox_size)
  y_reg, x_reg, z_reg = np.meshgrid(ls,ls,ls)
  #make grids into sets of data points
  warp_coords = np.stack((grid[2].flatten(), grid[1].flatten(), grid[0].flatten()), axis=1)
  #make regular grids into sets of data points
  reg_coords_x = x_reg.flatten()
  reg_coords_y = y_reg.flatten()
  reg_coords_z = z_reg.flatten()
  reg_coords = np.stack((reg_coords_x, reg_coords_y, reg_coords_z), axis=1)
  interp_x = LinearNDInterpolator(warp_coords, reg_coords_x)
  interp_y = LinearNDInterpolator(warp_coords, reg_coords_y)
  interp_z = LinearNDInterpolator(warp_coords, reg_coords_z)
  #interpolate the irregular grid by the regular grid to get the forwardwarp foir each dimension
  x_frwarp = interp_x(reg_coords)
  y_frwarp = interp_y(reg_coords)
  z_frwarp = interp_z(reg_coords)
  #reshape the forward warp into 32x32x32
  x_frwarp = x_frwarp.reshape((32,32,32))
  y_frwarp = y_frwarp.reshape((32,32,32))
  z_frwarp = z_frwarp.reshape((32,32,32))
  for_grid = np.stack((x_frwarp, y_frwarp, z_frwarp), axis=0)
  return for_grid
