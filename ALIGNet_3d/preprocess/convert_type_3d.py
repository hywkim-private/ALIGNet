import torch 
from pathlib import Path
import pytorch3d as p3d
from torch.utils.data import random_split, Dataset, Subset
from pytorch3d import structures
from pytorch3d.ops import sample_points_from_meshes
from . import voxelize, augment_3d, datasets


    
  

#get the minimum range of a points
#pointcloud: shape(minibatch, num_clouds, 3)
def get_min_range(mesh):
  points = mesh.verts_list()
  min_ = []
  max_ = []
  for point in points:
    max_d = torch.max(point[:,0])
    max_w = torch.max(point[:,1])
    max_h = torch.max(point[:,2])
    min_d = torch.min(point[:,0])
    min_w = torch.min(point[:,1])
    min_h = torch.min(point[:,2])
    max_cloud = max([max_d, max_w, max_h])
    min_cloud = min([min_d, min_w, min_h])
    max_.append(max_cloud)
    min_.append(min_cloud)
  max_ = max(max_)
  min_ = min(min_)
  range_ = (min_, max_)
  return range_


#given the min_max range of coordinates for a vertice, scale the point cloud as close as possible to the range (-1, 1)
#input
#coord_range: (min, max) range of coordinates of the mesh
#vert: Pytorch3d Mesh structure
#returns:
#None: in-place opoerator 
def scale_vert(coord_range, mesh):
  min_, max_ = coord_range
  left_edge = abs(min_)
  right_edge = abs(max_)
  #the minimum of the two edge will be used to determine the scale factor
  scale_edge = min(left_edge, right_edge)
  #scale all coordinates by a factor that will maximally stretch the farthest-reaching axis
  scale_factor = 1 / scale_edge
  scale_factor = scale_factor.item()
  mesh.scale_verts_(scale_factor)

  
#given dataset of meshes, sample data into pointclouds
#if mesh is False, we are using the pointcloud 
def PointCloud(mesh, num_samples):
  samples = sample_points_from_meshes(mesh, num_samples=num_samples)
  features=torch.zeros(len(samples), num_samples, 5)
  pointcloud = structures.Pointclouds(samples, features=features)
  return pointcloud 


#given dataset of pointclouds, voxelize data
#pointcloud: shape(minibatch, 3)
#voxel_num: int - the number of voxels in axis
#if mesh is False, we are using the pointcloud 
def Voxel(pointcloud, voxel_num):
  voxel = voxelize.voxelize_pointclouds(pointcloud,(voxel_num, voxel_num, voxel_num))
  return voxel


#load the mesh from a given location, save both the pointcloud and voxel representation of the model
class Compound_Data():
  #load the mesh data
  def __init__(self, mesh):
    self.mesh = mesh
    self.pointcloud = None
    self.voxel = None
  #a replicate of the torch.to() functionality 
  #apply torch.to() to all the Tensors in the compound class
  def to(self, device):
    self.mesh.to(device)
    if self.pointcloud is not None:
      self.pointcloud.to(device)
    if self.voxel is not None:
      self.voxel.to(device)
    
  def scale_mesh(self):
    min_range = get_min_range(self.mesh)
    scale_vert(min_range, self.mesh)

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
  def __init__(self, data, data_sec=None):
    self.data = data
    self.is_sec = False
    if data_sec:
      self.is_sec = True
      self.data_sec = data_sec
  def __getitem__(self, index):
    if self.is_sec:
      x = self.data[index]
      y = self.data_sec[index]
      return x,y
    else:
      x = self.data[index]
      return x
  def __len__(self):
    return len(self.data)
  def collate_fn(self, batch):
    if self.is_sec:
      x,y = zip(*batch)
      x = list(x)
      x = torch.stack(x, dim=0)
      y = list(y)
      return x,y
    else:
      x = batch
      x = list(x)
      x = torch.stack(x, dim=0)
      return x
     

#input
#A dataset to perform the operation of expanding the size of the dataset
#if mesh is not None, the dataset will return a tuple of (voxel, mesh) (default:False)
class Expand(Dataset): 
  def __init__(self, voxel, expand_size=None, mesh=None):
    self.voxel = voxel
    self.mesh = mesh
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
      #get the random index with which to fill in the remaining length
      tail_idx, _ = datasets.sample_index(tail_length, len(voxel)-tail_length, len(voxel))
      #fill in the tail of the voxel dataset
      voxel = Subset(voxel, tail_idx)
      vox_list.append(voxel)
      #the case for when we are also returning mesh
      if self.is_mesh:
        if not len(mesh) == len(tail_idx):
          mesh = mesh.__getitem__(tail_idx)
        mesh_list.append(mesh)
    self.vox_list = torch.utils.data.ConcatDataset(vox_list)
    #only initialize mesh_list if specified by tehmesh param
    if self.is_mesh:
      self.mesh_list = torch.utils.data.ConcatDataset(mesh_list)


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
      
      
#given source and target dataset, return a dict of augmented datasets
#if val_set is set to True, the loader returns one (identical) element
#for 3d, we will only support random mask operation
#input:
#tar: target dataset of voxels
#batch_size: size of a single batch
#size: ? 
#val_set: specifies whether it is a validation set => if so, different return values and augmentation routines
#augment_time: number of times to augment the dataset => length of the return value will be augment_times * len(tar)
#pointcloud: (only active for val_set=True) if pointcloud is given, also augment the pointcloud
#augment times should always be more than or equal to 1 
class Augment_3d(Dataset):
  def __init__(self, tar, batch_size, vox_size, mask_size, val_set=False, augment_times=1, pointcloud=None):
    self.is_pt = False if pointcloud is None else True
    self.val_set = val_set
    self.batch_size = batch_size
    self.augment_times = augment_times
    self.is_aug = False if augment_times == 0 else True
    #first augment the original tar_list to match the returning batch size, then perform augmentation
    tar_list = []
    for i in range(self.augment_times):
      tar_list.append(tar)
    #if is_pt is True, also get the pointcloud representation 
    if self.is_pt:
      pt_list = []
      for i in range(self.augment_times):
        pt_list.append(pointcloud)
      if self.is_aug:
        self.pt_list = torch.utils.data.ConcatDataset(pt_list)
    if self.is_aug:
      self.tar_list = torch.utils.data.ConcatDataset(tar_list)
      self.aug_list = torch.utils.data.ConcatDataset(tar_list)
    self.vox_size = vox_size
    self.mask_size = mask_size

  def __getitem__(self, index):
    #return augmented list when val_set is set to true
    if self.val_set:
      #if val_set  and is_pt, return augmented pointclouds in addition to voxels
      if self.is_pt:
        x = self.aug_list[index]
        y = self.pt_list[index]
        return x, y
      else:
        x = self.aug_list[index]
        return x
    #the case for when the set is for training   
    #return both tar_list and aug_list
    x = self.tar_list[index]
    y = self.aug_list[index]
    return x,y

  def __len__(self):
    return len(self.tar_list)

  #perform the mask operation on a batch 
  #return both the augmented voxel and the original voxe
  #if pt_batch is given, also augment the pointclouds 
  #input shape: batchx NxDxWxH
  #TODO: PERHAPS UTILIZE THE GET_DISAPPEARED FUNCTIONALITY OF RANDOM_MASK_3D
  def mask(self, vox_batch, pt_batch=None):
    #if pt_batch exists, pass in the batch parameter as a tuple
    if not pt_batch == None: 
      aug_vox, aug_pt = augment_3d.random_mask_3d((vox_batch, pt_batch), self.vox_size, self.mask_size, square=True)
      return aug_vox, aug_pt
    else:
      aug_vox = augment_3d.random_mask_3d(vox_batch, self.vox_size, self.mask_size, square=True)
      return aug_vox

  def collate_fn(self, batch):
    #if val_set, the batch will be a single augmented batch
    if self.val_set:
      if self.is_pt:
        vox_batch, pt_batch = zip(*batch)
        vox_batch = torch.stack(list(vox_batch), dim=0)
        if self.is_aug:
          pt_batch = list(pt_batch)
          pt_list = []
          #concatenate the batch into type Pointcloud
          for pt in pt_batch:
            pt_list.append(torch.cat(pt.points_list(), dim=0))
          pt_ts = torch.stack(pt_list, dim=0)
          #pointcloud datatype
          pt_batch = structures.Pointclouds(pt_ts)
          vox_batch, pt_batch = self.mask(vox_batch, pt_batch)
          return vox_batch, pt_batch
      else:
        vox_batch = torch.stack(list(batch), dim=0)
        if self.is_aug:
          vox_batch = self.mask(vox_batch) 
          return vox_batch
    else:
      aug_batch, tar_batch = zip(*batch)
      aug_batch = torch.stack(list(aug_batch), dim=0)
      tar_batch = torch.stack(list(tar_batch), dim=0)
      #the batch will be of shape batchx3xDxNxWxH)
      if self.is_aug:
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
