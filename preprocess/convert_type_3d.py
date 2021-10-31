import torch 
from pathlib import Path
import pytorch3d as p3d
from torch.utils.data import random_split, Dataset
from pytorch3d import structures
from pytorch3d.ops import sample_points_from_meshes
from . import voxelize, augment_3d

#given dataset of meshes, sample data into pointclouds
#if mesh is False, we are using the pointcloud 
def PointCloud(mesh, num_samples, device):
  samples = sample_points_from_meshes(mesh, num_samples=num_samples)
  features=torch.zeros(len(samples), num_samples, 5).to(device)
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
  def __init__(self, mesh, device):
    self.mesh = mesh
    print(f"Compound_Data: dtype mesh {mesh}")
    self.pointcloud = None
    self.voxel = None
    self.device = device
  def get_pointcloud(self, num_samples):
    self.pointcloud = PointCloud(self.mesh, num_samples, self.device)
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

#input
#A dataset to perform the operation of expanding the size of the dataset
#if mesh is not None, the dataset will return a tuple of (voxel, mesh) (default:False)
class Expand(Dataset): 
  def __init__(self, voxel, expand_size=None, mesh=None):
    self.voxel = voxel
    self.mesh = mesh
    print(f"Expand: type of mesh {self.mesh}")
    print(f"Expand: type of voxel {self.voxel}")

    #specifies if mes is to be returned or not
    self.is_mesh = True if mesh is not None else None
    self.expand_size = expand_size
    vox_list = []
    mesh_list = []
    while len(voxel)*len(vox_list) + len(voxel) < expand_size:
      vox_list.append(voxel)
      #the case for when we are also returning mesh
      if self.is_mesh: 
        mesh_list.append(mesh)
    #handle for the remaining number of img to fullfill the expand_size
    if not len(voxel)*len(vox_list) == expand_size:
      tail_length = expand_size - len(voxel) * len(vox_list) 
      voxel, dump = random_split(voxel, [tail_length, len(voxel)-tail_length])
      vox_list.append(voxel)
      #the case for when we are also returning mesh
      if self.is_mesh:
        mesh, dump = random_split(mesh, [tail_length, len(mesh)-tail_length])
        mesh_list.append(mesh)
    self.vox_list = torch.utils.data.ConcatDataset(vox_list)
    #only initialize mesh_list if specified by tehmesh param
    if self.is_mesh:
      self.mesh_list = torch.utils.data.ConcatDataset(mesh_list)
      print(f"Expand: type of mesh_list {self.mesh_list}")
      print(f"Expand: type of vox_list {self.vox_list}")

  def __getitem__(self, index):
    x = self.vox_list[index]
    #return both voxel and mesh representations if mesh is not None
    if self.is_mesh:
      y = self.mesh_list[index]
      return x, y
    #else, we just return the voxel representation 
    else:
      return x
  def __len__(self):
    return len(self.vox_list)
    
  def collate_fn(self, batch):
    if self.is_mesh:
      vox, mesh = zip(*batch)
      vox = torch.stack(list(vox), dim=0)
      #since we can't use torch.stack for Meshes datatype, we will return a batch of list
      mesh = list(mesh)
      return vox, mesh
    else: 
      vox = torch.stack(batch,dim=0)
      return vox
#given source and target img dataset, return a dict of augmented datasets
#if val_set is set to True, the loader returns one (identical) element
#for 3d, we will only support random mask operation
class Augment_3d(Dataset):
  def __init__(self, tar, batch_size, size, val_set=False, augment_times=0):
    self.val_set = val_set
    self.batch_size = batch_size
    self.augment_times = augment_times
    #first augment the original tar_list to match the returning batch size, then perform augmentation
    tar_list = []
    for i in range(self.augment_times):
      tar_list.append(tar)
    self.tar_list = torch.utils.data.ConcatDataset(tar_list)
    self.aug_list = torch.utils.data.ConcatDataset(tar_list)
    self.size = size

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
    aug_vx = augment_3d.random_mask_3d(aug_batch, self.size, (10, 40), square=True)
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





