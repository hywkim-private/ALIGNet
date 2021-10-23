import torch 
from pathlib import Path
import pytorch3d as p3d
from torch.utils.data import random_split
from pytorch3d import ops, structures
from preprocess import voxellize, io_3d, augment_3d

#given dataset of meshes, sample data into pointclouds
#if mesh is False, we are using the pointcloud 
def PointCloud(mesh, num_samples):
  samples = ops.sample_points_from_meshes(mesh, num_samples=num_samples)
  features=torch.zeros(len(samples), num_samples, 5).to(DEVICE)
  cloud = structures.Pointclouds(samples, features=features)
  #the points in the form of (minibatch, 3)
  points = torch.stack(cloud.points_list(), axis=0)
  return points


#given dataset of pointclouds, voxelize data
#pointcloud: shape(minibatch, 3)
#voxel_num: int - the number of voxels in axis
#if mesh is False, we are using the pointcloud 
def Voxel(pointcloud, voxel_num):
  coord_range = voxelize.get_min_range(pointcloud)
  voxel = voxelize.voxelize_pointclouds(pointcloud, [coord_range, coord_range, coord_range], (voxel_num, voxel_num, voxel_num))
  return voxel


#load the mesh from a given location, save both the pointcloud and voxel representation of the model
class Compound_Data():
  #load the mesh data
  def __init__(self, filepath, sample_index=None):
    mesh = Load_Mesh(filepath, sample_index).to(DEVICE)
    self.mesh = mesh
    self.pointcloud = None
    self.voxel = None
  def get_pointcloud(self, num_samples):
    self.pointcloud = PointCloud(self.mesh, num_samples)
  def get_voxel(self, voxel_num):
    if not self.pointcloud == None:
      self.voxel = Voxel(self.pointcloud, voxel_num)
    else:
      print("Error: Must initialize pointcloud first before voxels")
      return
  
  
#basic class for creating dataset out of a given input
class DS(Dataset):
  def __init__(self, data):
    self.data = data
  def __getitem__(self, index):
    x = self.data[index]
    return x
  def __len__(self):
    return len(self.data)


#A dataset to perform the operation of expanding the size of the dataset
#SAME AS 2D
class Expand(Dataset):
  def __init__(self, tar_img, expand_size=None):
    self.tar_img = tar_img
    self.expand_size = expand_size
    tar_list = []
    while len(tar_img)*len(tar_list) + len(tar_img) < expand_size:
      tar_list.append(tar_img)
    #handle for the remaining number of img to fullfill the expand_size
    if not len(tar_img)*len(tar_list) == expand_size:
      tail_length = expand_size - len(tar_img) * len(tar_list) 
      tar_img, dump = random_split(tar_img, [tail_length, len(tar_img)-tail_length])
      tar_list.append(tar_img)
    self.tar_list = torch.utils.data.ConcatDataset(tar_list)
  def __getitem__(self, index):
    x = self.tar_list[index]
    return x
  def __len__(self):
    return len(self.tar_list)

#given source and target img dataset, return a dict of augmented datasets
#if val_set is set to True, the loader returns one (identical) element
#for 3d, we will only support random mask operation
class Augment_3d(Dataset):
  def __init__(self, tar_img, batch_size, im_size, val_set=False, augment_times=0):
    self.val_set = val_set
    self.batch_size = batch_size
    self.augment_times = augment_times
    #first augment the original tar_list to match the returning batch size, then perform augmentation
    tar_list = []
    for i in range(self.augment_times):
      tar_list.append(tar_img)
    self.tar_list = torch.utils.data.ConcatDataset(tar_list)
    self.aug_list = torch.utils.data.ConcatDataset(tar_list)
    self.im_size = im_size

  def __getitem__(self, index):
    #return augmented list when val_set is set to true
    if self.val_set:
      y = self.aug_list[index]
      return y
    #if not, return both tar_list and aug_list
    x = self.tar_list[index]
    y = self.aug_list[index]
    return x,y

  def __len__(self):
    return len(self.tar_list)

  #perform the mask operation on a batch 
  #return both the augmented voxel and the original voxel
  #input shape: batchx NxDxWxH
  def mask(self, aug_batch):
    aug_vx = augment_3d.random_mask_3d(aug_batch, self.im_size, (10, 40), square=True)
    return aug_vx

  def collate_fn(self, batch):
    #if val_set, the batch will be a single augmented batch
    if self.val_set:
      batch = torch.stack(list(batch), dim=0)
      if self.augment_times > 0:
        aug_batch = self.mask(batch) 
      return aug_batch
    aug_batch, tar_batch = zip(*batch)
    aug_batch = torch.stack(list(aug_batch), dim=0)
    tar_batch = torch.stack(list(tar_batch), dim=0)
    #the batch will be of shape batchx3xDxNxWxH)
    if self.augment_times > 0:
      aug_batch = self.mask(aug_batch) 
    return aug_batch, tar_batch

#helper function for the class pre_augment
def pre_augment_(data_loader, val_set = False, batch_size=0, shuffle=False):
  shuffle_ = shuffle
  batch_size_ = batch_size
  ds_list_aug = []
  ds_list_tar = []
  if val_set:
    for aug_batch in data_loader:
      ds_list_aug.append(aug_batch)
    aug_ds = torch.utils.data.ConcatDataset(ds_list_aug)
    return aug_ds
  else:
    for aug_batch, tar_batch in data_loader:
      ds_list_aug.append(aug_batch)
      ds_list_tar.append(tar_batch)
    aug_ds=torch.utils.data.ConcatDataset(ds_list_aug)
    tar_ds=torch.utils.data.ConcatDataset(ds_list_tar)
    return aug_ds, tar_ds

#inorder to reduce time for augmentation, define a wrapper function that takes an "augmentable" dataloader, pre-process all the augmenting sequences, and returns a loader of augmented ds
#SAME AS 2D
class pre_augment(Dataset):
  def __init__(self, data_loader, val_set=False, batch_size=0, shuffle=False):
    batch_size_ = batch_size
    shuffle_ = shuffle
    self.val_set = val_set
    if val_set:
      self.aug_ds = pre_augment_(data_loader, val_set, batch_size_, shuffle_)
    else:
      self.aug_ds, self.tar_ds = pre_augment_(data_loader, val_set, batch_size_, shuffle_)
  def __len__(self):
    return len(self.aug_ds)
  def __getitem__(self, index):
    if self.val_set:
      return self.aug_ds[index]
    else: 
      return self.aug_ds[index], self.tar_ds[index]





