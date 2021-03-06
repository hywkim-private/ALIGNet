import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from . import convert_type_3d as conv
#Logic for preserving the voxel representation
#We only need to preserve voxel representations for when the data set is a valid and source dataaset
#otherwise, we will not save the voxel representation -- this functionality may change in future updates


#given train size and val_size, return a random list of indexes for train and val set samples
def sample_index(train_size, val_size, total_num):
  pool= np.arange(total_num)
  index_train = np.random.choice(pool, size=train_size, replace=False)
  pool = np.delete(pool, index_train)
  index_val = np.random.choice(pool, size=val_size, replace=False)
  index_train = np.sort(index_train)
  index_val = np.sort(index_val)
  return index_train, index_val

#given an index list an a size to sample, return a list of sampled indexes
def sample_index_list(index_list, sample_size):
  #return error if sample size is bigger than the len of the index list
  if sample_size > len(index_list):
    print(f"sample_index_list: Error sample size {sample_size} cannot be bigger than index list length {len(index_list)}")
    exit()
  #define the index from whichj to sample 
  sample_index = np.arange(len(index_list))
  sample_index = np.random.choice(sample_index, size=sample_size, replace=True)
  new_list = index_list[sample_index]
  new_list = np.sort(new_list)
  return new_list

#helper function for converting mesh datatype samples into Tensors of Meshes
def mesh_to_tensor(meshes, idx):
  mesh_list = []
  for i in idx:
    mesh = meshes.__getitem__(i)
    mesh_list.append(mesh)
  mesh_ts = ConcatDataset(mesh_list)
  return mesh_ts
  
#helper function for converting pt datatypes into Tensors of PointClouds
def pt_to_tensor(pt, idx):
  pt_list = []
  for i in idx:
    p = pt.__getitem__(i)
    pt_list.append(pt)
  pt_ts = ConcatDataset(pt_list)
  return pt_ts


#convert the types of datasets
#helper function for aug_datasets_3d
#we split this functionality to gurantee manual augmentation of data on same randomly-split datasets
#returns: a set of train and valid datasets of meshes
def get_datasets_3d(tr, val, vox_size, pt_sample):
  train_set = conv.Compound_Data(tr)
  val_set = conv.Compound_Data(val)
  train_set.scale_mesh()
  train_set.get_pointcloud(pt_sample)
  train_set.get_voxel(vox_size)
  val_set.scale_mesh()
  val_set.get_pointcloud(pt_sample)
  val_set.get_voxel(vox_size)
  return train_set, val_set


#augment datasets according to the parameters
#dataset-the data to augment(CompoundData)
#settype-whether it is a train or valid data (o for train 1 for valid)
#split_proportion-the proportion of target data (souce data proportion = 1 - split_proportion)
#batch_size-size of batch
#vox_size-size of a voxel along a single axis (int)
#augment_times-how many times to augment train dataset
#get_src_mesh-get the original mesh representation of the src dataset (only applies to the src dataset)
#get_tar_pt=get the original pointcloud representation of the tar datast (only applies to tar dataset)
#(even if ds is valid set, we need to optionalize get_mesh since the set may be used soley for loss-checking)
def aug_datasets_3d(dataset, settype, split_proportion, batch_size, vox_size, augment_times, mask_size, val_sample=None, get_src_mesh=False, get_tar_pt=False):
  #parameter that denotes whether or not we should augment data
  aug = False if augment_times <= 0 else True
  voxel = dataset.voxel
  data_size = len(voxel)
  tar_idx, src_idx = sample_index(int(data_size*split_proportion), data_size - int(data_size*split_proportion), data_size)
  #if val_sample is set, further reduce the sampling dataset by the val_sample value 
  if val_sample is not None:
    tar_idx = sample_index_list(tar_idx, val_sample)
    src_idx = sample_index_list(src_idx, val_sample)
  #the Mesh.__getitem__ doesn't support np type index inputs ==> list works
  src_idx = src_idx.tolist()
  tar_idx = tar_idx.tolist()
  tar_vox = Subset(voxel, tar_idx)
  src_vox = Subset(voxel, src_idx)
  #augment the valid dataset
  #the main difference is that val ds holds the original mesh representation of the src datatset
  if settype == 1:
    #boolean parameters to check if dataset has been augmented
    tar_init = False
    src_init = False
    if get_tar_pt:
      pt = dataset.pointcloud
      tar_pt = pt.__getitem__(tar_idx)
      if aug:
        tar = conv.Augment_3d(tar_vox, batch_size, vox_size,mask_size, val_set=True, augment_times=augment_times, pointcloud=tar_pt)
      #when aug is not True =>< return a vanila dataset
      else:
        tar = conv.DS(tar_vox, tar_pt)
      tar_init = True
    #if it requires no pointcloud representations
    if tar_init == False:
      if aug:
        tar = conv.Augment_3d(tar_vox, batch_size, vox_size, mask_size, val_set=True, augment_times=augment_times)
      else:
        tar = conv.DS(tar_vox)
    #the expand type of src will return both voxel and  the mesh if it is a valid dataset
    #in this case src is a dataset that returns both src_voxel and src_mesh
    if get_src_mesh:
      src_mesh = dataset.mesh
      #get the src mesh and the tar mesh
      #we need to convert mesh ds to tensor Dataset in order to pass it on as a Dataset module 
      src_mesh = src_mesh.__getitem__(src_idx)
      #src will be a dataloader that returns both target and original src mesh
      if aug: 
        src = conv.Expand(src_vox, len(tar), mesh = src_mesh)
      #return vanila dataset if augmenttimes is 0
      else:
        src = conv.DS(src_vox, src_mesh)
      src_init = True
    if src_init == False:
      if aug:
        src = conv.Expand(src_vox, len(tar))
      #return vanila dataset if augmenttimes is 0
      else:
        src = conv.DS(src_vox)
    return tar, src
  else:
    if aug:
      tar = conv.Augment_3d(tar_vox, batch_size, vox_size, mask_size,augment_times=augment_times)
      src = conv.Expand(src_vox, len(tar))
    else:
      tar = conv.DS(tar_vox)
      src = conv.DS(src_vox)
    return tar, src


#given dataset, return the appropriate dataloader
#SAME as 2d
def get_dataloader(ds, batch_size, augment=False, shuffle=False):
  SHUFFLE = shuffle 
  dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn, shuffle=SHUFFLE)
  return dl
  
#given dataset, return an appropriate distributed dataloader
def get_dataloader_parallel(ds, batch_size, shuffle=False):
  #define the sampler for splitting datasets
  sampler = DistributedSampler(ds,
                               shuffle=shuffle, 
                               seed=42)
  data_loader = DataLoader(ds,
                          batch_size=batch_size,
                          shuffle=False,  
                          sampler=sampler,
                          pin_memory=True)
  return data_loader

#the param get_mesh specifies whether or not to get the original mesh representation of the src dataset
def get_val_dl_3d(val, split_proportion, batch_size, vox_size, augment_times, mask_size, val_sample = None, get_src_mesh=False, get_tar_pt=False):
  augment = False if augment_times <= 0 else True
  val_tar_ds, val_src_ds = aug_datasets_3d(
    val, 1, split_proportion, batch_size, vox_size, augment_times, mask_size, val_sample = val_sample, get_src_mesh = get_src_mesh, get_tar_pt=get_tar_pt)
  val_tar_dl = get_dataloader(val_tar_ds, batch_size, augment=augment, shuffle=True)
  val_src_dl = get_dataloader(val_src_ds, batch_size, shuffle=True)
  return val_tar_dl, val_src_dl