#Helper functions for voxelization
import torch
import math

#get the minimum range of a pointcloud
#pointcloud: shape(minibatch, num_clouds, 3)
def get_min_range(pointcloud):
  min_ = []
  max_ = []
  for cloud in pointcloud:
    max_d = torch.max(cloud[:,0])
    max_w = torch.max(cloud[:,1])
    max_h = torch.max(cloud[:,2])
    min_d = torch.min(cloud[:,0])
    min_w = torch.min(cloud[:,1])
    min_h = torch.min(cloud[:,2])
    max_cloud = max([max_d, max_w, max_h])
    min_cloud = min([min_d, min_w, min_h])
    max_.append(max_cloud)
    min_.append(min_cloud)
  max_ = max(max_)
  min_ = min(min_)
  range_ = (min_, max_)
  return range_

#convert a 1d coordinate system to an index value in voxelized representations 
#range: type-tuple, shape - (min, max), the range of coordinates wherein the pointclouds lie
#num_cell: shape(depth, width, height) the number of cells on each axis 
#coordinate: the coordinate of the original system, dtype: float
#return: index 
def convert_coordinate_to_index(range, num_cell, coordinate):
  min, max = range
  #total length of the original coordinate system
  length = max - min
  #the index value for the coordinate
  index = (coordinate - min) / (length / num_cell)
  index = math.floor(index)
  #check for the case when index = num_cell (this error depends on the env running the script)
  if index == num_cell:
    index -= index
  return index 

#wherever one or more pointclouds land, the corresponding voxel will be set to 1, else 0
#pointcloud: shape(minibatch, num_clouds, 3)
#range: type-list of tuples, shape [(min,max), (min,max), (min,max)] - the range of coordinates wherein the pointclouds lie
#voxel_num: shape(depth, width, height) the number of voxels on each axis 
#return voxel of shape(len(pointcloud), voxel_num_x, voxel_num_y, voxel_num_z)
def voxelize_pointclouds(pointcloud, coord_range, voxel_num):
  voxel_num_z, voxel_num_x, voxel_num_y = voxel_num 
  #convert the pointcloud coordinate system to a corresponding index in the voxel grids
  volume_list = []
  for cloud in pointcloud:
    #initialize the volume where voxels are contained
    #tensor of shape(voxel_num, voxel_num, voxel_num)
    volume = torch.zeros([voxel_num_x, voxel_num_y, voxel_num_z])
    max_len = max(voxel_num_x, voxel_num_y, voxel_num_z)
    for i in range(len(cloud)):
      #for each point in pointclouds
      point = cloud[i]
      #print(point)
      #locate the index of the coordinate in voxel grids
      coord_x = convert_coordinate_to_index(coord_range[0], voxel_num_x, point[0])  
      coord_y = convert_coordinate_to_index(coord_range[2], voxel_num_y, point[2])
      coord_z = convert_coordinate_to_index(coord_range[1], voxel_num_z, point[1])  
      #set the corresponding voxel value to 1 
      volume[coord_x, coord_y, coord_z] = 1
    volume_list.append(volume)
  volume_tensor = torch.stack(volume_list,axis=0)
  return volume_tensor


