#DATA AUGMENTATION ROUTINES
import torch 
import torchvision
import numpy as np
from pytorch3d.structures import Pointclouds


#TODO: SPLIT THIS FUNCTION IN TWO => FOR PT AND VOXEL
#return target image as masked by random size and locations
#mask_size can either be integer or tuple, specifying the range of random mask sizes 
#data: can either be Voxel datatype or tuple (Voxel, Pointcloud) 
#**IMPORTANT**
#receives pointcloud datastructure and returns poincloud datastructure
#get_disappeared: specifies whether or not to retrieve the deleted points (for visualization purposes)
#if get_disappeared is set, returns masked_voxel, masked_pt, disappeared_points
#input: BxDxWxH
def random_mask_3d(data, size, mask_size, square=True, recursion_counter=0, get_disappeared = False):
  #if data type is tuple, mask both voxel and pointcloud data
  if type(data) is tuple:
    voxel, pt = data
    masked_pt = pt.clone()
    #we will manipulate is points_list and return Pointscloud datastructure constructed with this list
    pt_ts = torch.stack(masked_pt.points_list(), dim=0)
    is_pt = True
  else:
    voxel = data
    is_pt = False
  disappeared_list =[]
  #the central point from which to create mask
  center = int(size / 2)
  #direction along which to apply the mask
  #if 0, direction of decreasing index, if 1 increasing index
  direction_x = np.random.random_integers(0,1)
  direction_y = np.random.random_integers(0,1)
  direction_z = np.random.random_integers(0,1)
  if recursion_counter > 10: 
    return voxel
  masked_voxel = voxel.clone()
  i = 0
  reapply_counter = 0 
  while i < len(masked_voxel):
    #define starting coordinates as offset random +- amounts from the center
    start_coordinate_x = center + np.random.randint(-int(size/3), int(size/3))
    start_coordinate_y = center + np.random.randint(-int(size/3), int(size/3))
    start_coordinate_z = center + np.random.randint(-int(size/3), int(size/3))
    mask_size_x=0
    mask_size_y=0
    mask_size_z=0
    if type(mask_size == 'tuple'):
      mask_size_low, mask_size_high = mask_size
      mask_size_x = np.random.randint(mask_size_low, mask_size_high)
      mask_size_y = mask_size_x if square else np.random.randint(mask_size_low, mask_size_high)
      mask_size_z = mask_size_x if square else np.random.randint(mask_size_low, mask_size_high)
    else:
      mask_size_x = mask_size
      mask_size_y = mask_size
      mask_size_x = mask_size
    #define the end coordinates for each axis
    if direction_x == 0:
      #when direction is 0, reverse start and end coordinates
      end_coordinate_x = start_coordinate_x
      start_coordinate_x = start_coordinate_x - mask_size_x if start_coordinate_x - mask_size_x > 0 else 0
    elif direction_x == 1:
      end_coordinate_x = start_coordinate_x + mask_size_x if start_coordinate_x + mask_size_x < size else size 
    if direction_y == 0:
      end_coordinate_y = start_coordinate_y
      start_coordinate_y = start_coordinate_y - mask_size_y if start_coordinate_y - mask_size_y > 0 else 0
    elif direction_y == 1:
      end_coordinate_y = start_coordinate_y + mask_size_y if start_coordinate_y + mask_size_y < size else size 
    if direction_z == 0:
      end_coordinate_z = start_coordinate_z
      start_coordinate_z = start_coordinate_z - mask_size_z if  start_coordinate_z - mask_size_z > 0 else 0
    elif direction_z == 1:
      end_coordinate_z =  start_coordinate_z + mask_size_z if start_coordinate_z + mask_size_z < size else size 
    masked_voxel[i, start_coordinate_x:end_coordinate_x, start_coordinate_y:end_coordinate_y,  start_coordinate_z:end_coordinate_z] = 0
      #if the masked_voxel is same as the original voxel, apply the random mask again
    if not torch.equal(masked_voxel[i], voxel[i]):
      i += 1  
    else:
      if reapply_counter >= 10:
        i += 1
        reapply_counter = 0
      else:
        reapply_counter += 1
    #if is_pt is True, apply the mask operation to pointcloud
    if is_pt:
      start_coordinate_x = (start_coordinate_x-(size/2)) / (size/2)
      start_coordinate_y = (start_coordinate_y-(size/2)) / (size/2)
      start_coordinate_z = (start_coordinate_z-(size/2)) / (size/2)
      end_coordinate_x = (end_coordinate_x-(size/2)) / (size/2)
      end_coordinate_y = (end_coordinate_y-(size/2)) / (size/2)
      end_coordinate_z = (end_coordinate_z-(size/2)) / (size/2)
      #identify the target pointcloud
      pt = pt_ts[i-1]
      #store the deleted idx from the verts_lists => use this list to delete faces
      deleted_idx = []
      zero_ts = torch.Tensor([[0,0,0]])
      disappeared_pts = []
      for idx, point in enumerate(pt):
        #delete the vertice if any of the coordinates are within the delete range
        #Logic: we need to keep the "padded" datastructure of pointcloud intact
        #Thus we will remove the selected "in-box" points and attatch a zero tensor at the very end
        if start_coordinate_x <= point[0] and point[0] <= end_coordinate_x: 
          if start_coordinate_y <= point[1] and point[1] <= end_coordinate_y:
            if start_coordinate_z <= point[2] and point[2] <= end_coordinate_z:
              disappeared = point.clone()
              temp_pt = torch.cat((pt_ts[i-1,:idx], pt_ts[i-1, idx+1:]))
              pt_ts[i-1] = torch.cat((temp_pt, zero_ts))
              if get_disappeared:
                disappeared_pts.append(disappeared)
      if get_disappeared:
        if len(disappeared_pts) == 0:
          disappeared_pts.append(zero_ts[0])
        disappeared = torch.stack(disappeared_pts, dim=0)
        disappeared_list.append(disappeared)
      
  if is_pt:
    masked_pt = Pointclouds(pt_ts)
    if get_disappeared:
      return masked_voxel, masked_pt, disappeared_list
    else:
      return masked_voxel, masked_pt
  else:
    return masked_voxel

  

  
