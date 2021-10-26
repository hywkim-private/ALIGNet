import numpy as np
from torch.utils.data import random_split, DataLoader
from . import convert_type_3d as conv

#given train size and val_size, return a random list of indexes for train and val set samples
def sample_index(train_size, val_size, total_num):
  pool= np.arange(total_num)
  index_train = np.random.choice(pool, size=train_size)
  pool = np.delete(pool, index_train)
  index_val = np.random.choice(pool, size=val_size)
  return index_train, index_val

#convert the types of datasets
#helper function for aug_datasets_3d
#we split this functionality to gurantee manual augmentation of data on same randomly-split datasets
#returns: a set of train and valid datasets of meshes
def get_datasets_3d(tr, val, vox_size, pt_sample, device):
  train_set = conv.Compound_Data(tr, device)
  val_set = conv.Compound_Data(val, device)
  train_set.get_pointcloud(pt_sample)
  train_set.get_voxel(vox_size)
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
def aug_datasets_3d(dataset, settype, split_proportion, batch_size, vox_size, augment_times):
  data_size = len(dataset)
  tar, src = random_split(
    dataset, [int(data_size*split_proportion), data_size - int(data_size*split_proportion)])
  if settype == 1:
    tar = conv.Augment_3d(tar, batch_size, vox_size, val_set=True, augment_times=augment_times)
  else:
    tar = conv.Augment_3d(tar, batch_size, vox_size, augment_times=augment_times)
  src = conv.Expand(src, len(tar))
  
  return tar, src
  
"""#given train and valid, return appropriatly augmented data for each set 
def aug_train_valid_3d(train_set, valid_set, split_proportion_tr, split_proportion_batch_size, vox_size, augment_times_tr, augment_times_src):
  #get the train dataset
  tr_tar, tr_src = augment_datasets_3d(train_set, 0, split_proportion, batch_size, vox_size, augment_times_tr)
  #the the valid dataset
  
  TRAIN_SIZE = len(train_set)
  VAL_SIZE = len(valid_set)
  Trainset_target, Trainset_source = random_split(
    train_set, [int(TRAIN_SIZE*config_3d.TARGET_PROPORTION), len(train_set) - int(TRAIN_SIZE*config_3d.TARGET_PROPORTION)])
  Validset_target, Validset_source = random_split(
    valid_set, [int(VAL_SIZE*config_3d.), len(valid_set) - int(VAL_SIZE*config_3d.TARGET_PROPORTION_VAL)])
  train_tar = conv.Augment_3d(Trainset_target, BATCH_SIZE, 32, augment_times = augment_times)
  valid_tar = conv.Augment_3d(Validset_target, BATCH_SIZE, 32, val_set=True, augment_times = 1)
  train_src = conv.Expand(Trainset_source, len(train_tar))
  valid_src = conv.Expand(Validset_source, len(valid_tar))
  return train_tar, train_src, valid_tar, valid_src """

#given dataset, return the appropriate dataloader
#SAME as 2d
def get_dataloader(ds, batch_size, augment=False, shuffle=False):
  SHUFFLE = shuffle 
  if augment:
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn, shuffle=SHUFFLE)
  else:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=SHUFFLE)  
  return dl

def get_val_dl_3d(val, split_proportion, batch_size, vox_size, augment_times):
  val_tar_ds, val_src_ds = aug_datasets_3d(
    val.voxel, 1, split_proportion, batch_size, vox_size, augment_times=augment_times)
  val_tar_dl = get_dataloader(val_tar_ds, batch_size, augment=True, shuffle=True)
  val_src_dl = get_dataloader(val_src_ds, batch_size, shuffle=True)
  return val_tar_dl, val_src_dl