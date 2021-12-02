#Helper functions for voxelization
import torch
import math


  
#DEPRECATED=>USE CONVERT_COORDINATE_TO_INDEX)
#convert a 1d coordinate system to an index value in voxelized representations 
#automatically fits the shape into voxel range
#range: type-tuple, shape - (min, max), the range of coordinates wherein the pointclouds lie
#num_cell: shape(depth, width, height) the number of cells on each axis 
#coordinate: the coordinate of the original system, dtype: float
#return: index 
def convert_coordinate_to_index_expand(coord_range, num_cell, coordinate):
  min_, max_ = coord_range
  #total length of the original coordinate system
  length = max_ - min_
  #the index value for the coordinate
  index = (coordinate - min) / (length / num_cell)
  index = math.floor(index)
  #check for the case when index = num_cell (this error depends on the env running the script)
  if index == num_cell:
    index -= index
  return index 
  

#convert a 1d coordinate system to an index value in voxelized representations 
#unlike convert_coordinate_to_index_expand, this function doesn't scale coordinates => simply turns (-1,1) coord to (0, num_cell)
#range: type-tuple, shape - (min, max), the range of coordinates wherein the pointclouds lie
#num_cell: shape(depth, width, height) the number of cells on each axis 
#coordinate: the coordinate of the original system, dtype: float
#return: index 
def convert_coordinate_to_index(num_cell, coordinate):
  #convert to (0,2) coord frame
  coord_frame = coordinate + 1
  scale_factor = num_cell / 2
  index = math.floor((coord_frame * scale_factor).item())
  return index 

#given a voxel of shape (x,y,z), fill in the "fully enclosed voxels"--or voxel points closed on all sides
def fill_enclosed_voxels(voxel, vox_num):
  for i in range(vox_num):
    for j in range(vox_num):
      #denotes whether or not the voxel indice is enclosed along the specified axis
      x_enclosed = False
      y_enclosed = False
      z_enclosed = False
      #denotes if the enclosure boundry of x has been found
      x_found = False
      y_found = False
      z_found = False 

      #k is the target index
      for k in range(vox_num):
        #if x_start coordinate is found and the voxel is 1, define the range as enclosed
        if x_enclosed == True and voxel[k,i,j] == 1:
          x_found = True
          x_end=k
        if y_enclosed == True and voxel[i,k,j] == 1:
          y_found = True
          y_end=k
        if z_enclosed == True and voxel[i,j,k] == 1:
          z_found = True 
          z_end=k
        #find the first instance of 1
        if x_enclosed == False and voxel[k,i,j] == 1:
          x_enclosed = True
          x_start=k
        if y_enclosed == False and voxel[i,k,j] == 1:
          y_enclosed = True
          y_start=k
        if z_enclosed == False and voxel[i,j,k] == 1:
          z_enclosed = True
          z_start=k
        #if the enclosure boundry is found => set all values within that boundry to 1 
        if x_found:
          voxel[x_start:x_end,i,j]=1
        if y_found:
          voxel[i,y_start:y_end,j]=1
        if z_found:
          voxel[i,j,z_start:z_end]=1
  return voxel
        

#wherever one or more pointclouds land, the corresponding voxel will be set to 1, else 0
#pointcloud: shape(minibatch, num_clouds, 3)
#range: type-list of tuples, shape [(min,max), (min,max), (min,max)] - the range of coordinates wherein the pointclouds lie
#voxel_num: shape(depth, width, height) the number of voxels on each axis 
#return voxel of shape(len(pointcloud), voxel_num_x, voxel_num_y, voxel_num_z)
def voxelize_pointclouds(pointcloud, voxel_num):
  voxel_num_z, voxel_num_x, voxel_num_y = voxel_num 
  pointcloud = pointcloud.points_list()
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
      #locate the index of the coordinate in voxel grids
      coord_x = convert_coordinate_to_index(voxel_num_x, point[0])  
      coord_y = convert_coordinate_to_index(voxel_num_y, point[1])
      coord_z = convert_coordinate_to_index(voxel_num_z, point[2])  
      #set the corresponding voxel value to i
      if coord_x == voxel_num_x:
        coord_x -= 1
      if coord_y == voxel_num_y:
        coord_y -= 1
      if coord_z == voxel_num_z:
        coord_z -= 1
      volume[coord_x, coord_y, coord_z] = 1
    #volume = fill_enclosed_voxels(volume, voxel_num_x)
    volume_list.append(volume)
  volume_tensor = torch.stack(volume_list,axis=0)
  return volume_tensor