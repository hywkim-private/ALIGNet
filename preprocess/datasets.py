import numpy as np
from torch.utils.data import random_split
from ALIGNet_3d import config_3d
from preprocess import convert_type_3d as conv
#given train size and val_size, return a random list of indexes for train and val set samples
def sample_index(train_size, val_size, total_num):
  pool= np.arange(total_num)
  index_train = np.random.choice(pool, size=train_size)
  pool = np.delete(pool, index_train)
  index_val = np.random.choice(pool, size=val_size)
  return index_train, index_val

#load the datasets and split accordingly to the global definition
#we split this functionality to gurantee manual augmentation of data on same randomly-split datasets
#returns: a set of train and valid datasets of meshes
def get_datasets_3d(file_path, train_size, val_size, total_data):
  train_index, val_index = sample_index(train_size, val_size, total_data)
  train_set = conv.Compound_Data(file_path, train_index)
  val_set = conv.Compound_Data(file_path, val_index)
  train_set.get_pointcloud(10000)
  train_set.get_voxel(32)
  val_set.get_pointcloud(10000)
  val_set.get_voxel(32)
  return train_set, val_set


#given train and valid, return appropriatly augmented data for each set 
def aug_datasets_3d(train_set, valid_set, augment_times=None):
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
  return train_tar, train_src, valid_tar, valid_src 

#given dataset, return the appropriate dataloader
#SAME as 2d
def get_dataloader(ds, augment=False, shuffle=False):
  SHUFFLE = shuffle 
  if augment:
    dl = DataLoader(ds, batch_size=config_3d.BATCH_SIZE, collate_fn=ds.collate_fn, shuffle=SHUFFLE)
  else:
    dl = DataLoader(ds, batch_size=config_3d.BATCH_SIZE, shuffle=SHUFFLE)  
  return dl

def get_val_dl_3d(tr, val):
  tr_tar_ds, tr_src_ds, val_tar_ds, val_src_ds = aug_datasets_3d(tr, val)
  val_tar_dl = get_dataloader(val_tar_ds, augment=True, shuffle=True)
  val_src_dl = get_dataloader(val_src_ds, shuffle=True)
  return val_tar_dl, val_src_dl