
#DATA AUGMENTATION ROUTINES
import torch 
import torchvision
import numpy as np
from imgaug import augmenters as iaa
import torch.nn as nn
import torch.nn.functional as F
import skimage


#return target image as masked by random size and locations
#mask_size can either be integer or tuple, specifying the range of random mask sizes
#input: BxWxH
def random_mask_2d(target_image, target_size, mask_size, square=True, recursion_counter=0 ):
  if recursion_counter > 10: 
    return target_image
  masked_image = target_image.copy()
  for i in range(len(masked_image)-1):
    x_coordinate = np.random.randint(target_size)
    y_coordinate = np.random.randint(target_size)
    mask_size_x=0
    mask_size_y=0
    if type(mask_size == 'tuple'):
      mask_size_low, mask_size_high = mask_size
      mask_size_x = np.random.randint(mask_size_low, mask_size_high)
      mask_size_y = mask_size_x if square else np.random.randint(mask_size_low, mask_size_high)
    else:
      mask_size_x = mask_size
      mask_size_y = mask_size 

    end_coordinate_x = x_coordinate + mask_size_x if x_coordinate + mask_size_x < target_size else target_size
    end_coordinate_y = y_coordinate + mask_size_y if y_coordinate + mask_size_y < target_size else target_size
    masked_image[i][x_coordinate:end_coordinate_x,y_coordinate:end_coordinate_y] = 0

  #if the masked_image is same as the target_image, recursively apply the function again
  if np.array_equal(masked_image, target_image):
    #print('masked_image identical to target: reapplying random_mask')
    #print(f'recursion_counter: {recursion_counter}')
    recursion_counter += 1 
    return random_mask_2d( target_image, target_size, mask_size, square, recursion_counter = recursion_counter)
  return masked_image

#given an NxM image, detected the minimum gap between the edge of the image and the image pixel
def min_gap(image):
  N, M = image.shape
  #search the gap along the x_axis
  min_index_left_x = 0
  min_index_right_x = N-1
  for i in range(N-1):
    left_sum_x = sum(image[:,i])
    right_sum_x = sum(image[:,N-i-1])
    left_x_update = False
    right_x_update = False
    if left_sum_x != 0 and not left_x_update:
      min_index_left_x = i 
      left_x_update = True
    if right_sum_x != 0 and not right_x_update:
      min_index_right_x = N - i 
      right_x_update = True
    if left_x_update and right_x_update:
      break
  #search the gap along the y_axis
  min_index_upper_y = 0
  min_index_lower_y = M-1
  for k in range(M-1):
    upper_sum_y = sum(image[k,:])
    lower_sum_y = sum(image[M-k-1,:])
    upper_y_update = False
    lower_y_update = False
    if upper_sum_y != 0 and not upper_y_update:
      min_index_upper_y = k 
      upper_y_update = True
    if lower_sum_y != 0 and not lower_y_update:
      min_index_lower_y = M - k 
      lower_y_update = True
    if upper_y_update and lower_y_update:
      break
    
  return min_index_left_x, min_index_right_x, min_index_upper_y, min_index_lower_y
  

#randomly stretch image vertically or/and horizontally
#the transformation will be an affine transformation
#the transformation can either stretch inwards or outwards
#input WxH
def random_stretch (target_image, vertical=False, horizontal=False, stretch_inwards=False):
    W, H = target_image.shape
    grid_diff = 2/(W-1)
    min_index_left_x, min_index_right_x, min_index_upper_y, min_index_lower_y = min_gap(target_image)
    #define the limit range to which we will stretch the image
    limit_left_x = np.random.choice(np.arange(0,min_index_left_x),1).item()
    limit_right_x = np.random.choice(np.arange(min_index_right_x,W),1).item()
    limit_upper_y = np.random.choice(np.arange(0,min_index_upper_y),1).item()
    limit_lower_y = np.random.choice(np.arange(min_index_lower_y,H),1).item()
    stretch_center_x = int((limit_left_x + limit_right_x) / 2)
    evencenter_x = True if (limit_left_x + limit_right_x) % 2 == 0 else False 
    stretch_center_y = int((limit_upper_y + limit_lower_y) / 2)
    evencenter_y = True if (limit_upper_y + limit_lower_y) % 2 == 0 else False
    #define the differential scale we will add to scale window
    stretch_scale_left_x = grid_diff * (min_index_left_x - limit_left_x) / (stretch_center_x - min_index_left_x - (1 if evencenter_x else 0))
    stretch_scale_right_x = grid_diff * (limit_right_x - min_index_right_x) / (min_index_right_x - stretch_center_x + 1)
    stretch_scale_upper_y = grid_diff * (min_index_upper_y - limit_upper_y) / (stretch_center_y - min_index_upper_y - (1 if evencenter_y else 0))
    stretch_scale_lower_y = grid_diff * (limit_lower_y - min_index_lower_y) /  (min_index_right_x - stretch_center_x + 1)
    
    
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1,1,W), torch.linspace(-1,1,H))
    target_image = target_image.unsqueeze(0).unsqueeze(0)
    if horizontal:
      """#apply the transformation to within the stretch window
      grid_x[:, limit_left_x : stretch_center_x - (min_index_left_x - limit_left_x) - (1 if evencenter_x else 0)] += stretch_scale_left_x * (min_index_left_x - limit_left_x)
      grid_x[:, stretch_center_x + (limit_right_x - min_index_right_x) + 1 : limit_right_x] -= stretch_scale_right_x * (limit_right_x - min_index_right_x)"""
    if vertical:
      start_upper_y = stretch_center_y - (min_index_upper_y - limit_upper_y)
      start_lower_y = stretch_center_y + (limit_lower_y - min_index_lower_y)
      for i in range(start_upper_y, limit_upper_y, -1):
        grid_y = grid_y.clone()
        grid_y[:, i] += stretch_scale_upper_y 
        grid = torch.stack([grid_x, grid_y])
        grid = grid.type(torch.FloatTensor).to(DEVICE)
        target_image = nn.functional.grid_sample(target_image.permute([1,0,2,3]).to(DEVICE), grid.unsqueeze(0).permute([0,2,3,1]))
        grid = grid.squeeze()
      for j in range(start_lower_y, limit_lower_y, 1):
        grid_y = grid_y.clone()
        grid_y[:, j] -= stretch_scale_lower_y 
        grid = torch.stack([grid_x, grid_y])
        grid = grid.type(torch.FloatTensor).to(DEVICE)
        target_image = nn.functional.grid_sample(target_image.permute([1,0,2,3]).to(DEVICE), grid.unsqueeze(0).permute([0,2,3,1]))
        grid = grid.squeeze()
    

    
    torch.set_printoptions(edgeitems=100)
    grid = grid.type(torch.FloatTensor).to(DEVICE)
    #perform the cumsum operation to restore the original grid from the differential grid
    return target_image


#given mask_size, target_size, stride, and target_image, create as many masked target images as possible and return its corresponding dataset
def mask_target_2d(mask_size, stride, target_size, target_image):
  #define parameters to fine-tune the masking operation
  remainder = target_size % mask_size
  max_iter = int(target_size / mask_size)

  #even_remainder denotes whether the remaining pixel after masking the input image is even (default=false)
  even_remainder = False
  if remainder % 2 == 0:
    even_remainder = True
  #convert target image to numpy in order to ease type conversions
  target_image = target_image.to(torch.device('cpu'))
  target_image = np.array(target_image)  
  starting_coordinate = [mask_size+int(remainder/2), mask_size+int(remainder/2)]
  #the current_coordinate denotes the upper-right-most coordinate of the mask 
  current_coordinate = [mask_size+int(remainder/2), mask_size+int(remainder/2)]
  #max_coordinate denotes the upper_right-most corner of the target image that the mask can reach
  max_coordinate = [target_size-int(remainder/2),target_size-int(remainder/2)]
  #dataset of masked images
  masked_arr = []
  #the first element of masked data will be the original image
  masked_arr.append(target_image.copy())
  #produce mask images until the mask reaches the upper_right corner of the target image (max_coordinate)
  while (current_coordinate[0] <= max_coordinate[0]) and (current_coordinate[1] <= max_coordinate[1]):
    #turn all areas within the masking target into zero
    masked_image = target_image.copy()
    masked_image[:,current_coordinate[0]-mask_size:current_coordinate[0],current_coordinate[1]-mask_size:current_coordinate[1]] = 0
    #add the masked image to data array--only if the masked image is different from the original image
    if not np.array_equal(masked_image, target_image):
      masked_arr.append(masked_image)
    #update current_coordinate along the x-axis
    current_coordinate[0] = current_coordinate[0] + stride
    #if the mask reaches the max of the x axis, move up to the y axis 
    if current_coordinate[0] >= max_coordinate[0]:
      current_coordinate[1] = current_coordinate[1] + stride
      current_coordinate[0] = starting_coordinate[0] 
  masked_arr = np.array(masked_arr)
  #just return the np object, not tensor
  return masked_arr
    


#return the pipeline of a transformation
#we will use albumentations library for this function
def random_augmentation(transform_no = 1):
  #define the transforamtion pipline 
  sometimes = lambda aug: iaa.Sometimes(0.5, aug)
  sometimes_more = lambda aug: iaa.Sometimes(0.8, aug)

  if transform_no == 1:
    transform = iaa.Sequential (
    [
    sometimes_more(iaa.OneOf([
                iaa.Affine(scale={'x':(0.5,1.5), 'y':(0.5, 1.5)}, 
                translate_percent={'x':(0.1, 0.2),'y':(0.1,0.2)},
                rotate=(-60,60),
                shear=(-16,16)),
                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)])),
    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    sometimes(iaa.Dropout(p=0.2))
    
    ])
    return transform

  if transform_no == 2:
    transform2 = iaa.Sequential (
    [
      sometimes_more(iaa.OneOf([
        iaa.Affine(scale={'x':(0.5,1.5), 'y':(0.5, 1.5)}, 
        translate_percent={'x':(0.1, 0.2),'y':(0.1,0.2)},
        rotate=(-60,60),
        shear=(-16,16)),
        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)]))
    ])
    return transform2
  #this model trains extensively on affine transformed images
  if transform_no == 3: 
    transform3 = iaa.Sequential (
    [
        iaa.Affine(scale={'x':(0.5,1.5), 'y':(0.5, 1.5)}, 
        translate_percent={'x':(0.1, 0.2),'y':(0.1,0.2)},
        rotate=(-360,360),
        shear=(-16,16))
    ])
    return transform3

