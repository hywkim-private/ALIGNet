#DATA AUGMENTATION ROUTINES
import torch 
import torchvision
import numpy as np

#TODO: FOR IMPROVEMENT OF RANDOM MASK, MAKE A CENTER-LOCATING FUNCTION WHICH FINDS THE MIN-MAX AVG OF
#SPECIFIC VOXELS ALONG EACH X, Y, Z AXIS

#return target image as masked by random size and locations
#mask_size can either be integer or tuple, specifying the range of random mask sizes
#input: BxDxWxH
def random_mask_3d(voxel, size, mask_size, square=True, recursion_counter=0):
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
  return masked_voxel

  

  
