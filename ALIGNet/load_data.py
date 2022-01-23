#DATA LOADING ROUTINES
import config
import cv2
import wget
import h5py 
import torch 
import numpy as np
import augment
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor
import torchvision
from zipfile import ZipFile
from pathlib import Path


#download training data from the specified url
def download_data(url, extract_path):
  filename = wget.download(url)
  zf = ZipFile(filename, 'r')
  zf.extractall(extract_path)
  zf.close()
  
#load raw plane datasets
def load_ds(path):
  tr = torch.load(path+'tr'+ '.pt')
  val = torch.load(path+'val'+ '.pt')
  return tr, val
  
#This class must be fine-tuned after completing the model, so as to be able to read from any h5 input formats
class Load_HDF5(Dataset):
  # if get_all == True, file_path must specifiy a directory, if not, it should specify a file
  # the class will crash if this requirement is not met
  def __init__(self, file_path, get_all = False, transform = None):
    self.path = file_path
    self.transform = transform
    #if filename is None, perform a recursive search over the file directory and retrieve all .h5 files
    p = Path(file_path)
    if get_all == True:
      files = sorted(p.glob('*.h5'))
      data_list = []
      for file in files:
        data_list.append(self.get_file(str(file)))
      self.data = data_list
    else:
      self.data = self.get_file(str(file))

  #given a filepath, return the image object
  def get_file(self, path):
    with h5py.File(path, 'r') as file:
      data = file['dep']
      data = data.get('view1')[:]
      return data
    
  def __getitem__(self, index):
    if len(self.data) <= index:
      print("__getitem__ERROR: index out of range")
      return 
    else:
      x = self.data[index]
      x = x.astype('float32')
      #check if the loaded image matches IMAGE_SIZE
      #if not, either downsample or upsample the image
      if self.transform:
        x = self.transform(x)
      return x
  def __len__(self):
    return len(self.data)

#A dataset to perform the operation of expanding the size of the dataset
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
#if val_set is set to True, the loader returns one element, not two
class Augment(Dataset):
  def __init__(self, tar_img, batch_size, im_size, val_set=False, augment_times=0, transform=False, random_mask=False, transform_no = 1):
    self.val_set = val_set
    self.transform = transform
    self.random_mask = random_mask
    self.batch_size = batch_size
    self.aug = augment.random_augmentation(transform_no)
    self.augment_times = augment_times
    self.augment = False if augment_times==0 else True 
    #first augment the original tar_list to match the returning batch size, then perform augmentation
    if not augment_times == 0:
      tar_list = []
      for i in range(self.augment_times):
        tar_list.append(tar_img)
      self.tar_list = torch.utils.data.ConcatDataset(tar_list)
      self.aug_list = torch.utils.data.ConcatDataset(tar_list)
      
    #if augment_times is 1
    else:
      self.tar_list = tar_img
      self.aug_list = tar_img
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

  #perform the deformation operation on batch of images
  #input shape: batchx NxWxH
  def deform(self, aug_batch):
    if not (self.augment_times == None):
      aug_im = self.aug.augment_images(images=aug_batch)
    return aug_im

  #perform the mask operation on a batch 
  #return both the augmented image and the original image 
  #input shape: batchx NxWxH
  def mask(self, aug_batch):
    tar_batch = aug_batch
    if not (self.random_mask == None):
      aug_im = augment.random_mask_2d(aug_batch, self.im_size, (50, 80), square=True)
    return aug_im
      

  def collate_fn(self, batch):
    #if val_set, the batch will be a single augmented batch
    if self.val_set:
      if self.aug:
        batch = np.concatenate([*batch])
        if self.transform and self.random_mask:
          batch = self.deform(batch)
          batch = self.mask(batch)  
        elif self.transform:
          batch = self.deform(batch)
        elif self.random_mask:
          batch = self.mask(batch)
        batch = torch.tensor(batch, dtype=torch.float32, requires_grad=True)
      if self.device:
        batch = batch.to(self.device)
      return batch
    aug_batch, tar_batch = zip(*batch)
    aug_batch = np.concatenate(list(aug_batch))
    tar_batch = np.concatenate(list(tar_batch))
    if self.aug:
      #the batch will be a batchx2xNxWxH)
      if self.transform and self.random_mask:
        aug_batch = self.deform(aug_batch)
        tar_batch = aug_batch.copy()
        aug_batch = self.mask(aug_batch)  
      elif self.transform:
        aug_batch = self.deform(aug_batch)
        tar_batch = aug_batch.copy()
      elif self.random_mask:
        aug_batch = self.mask(aug_batch) 
      aug_batch = torch.tensor(aug_batch, dtype=torch.float32, requires_grad=True)
      tar_batch = torch.tensor(tar_batch, dtype=torch.float32, requires_grad=True)
    if self.device:
      aug_batch = aug_batch.to(self.device)
      tar_batch = tar_batch.to(self.device)
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



#given train size and val_size, return a random list of indexes for train and val set samples
def sample_index(train_size, val_size, total_num):
  pool= np.arange(total_num)
  index_train = np.random.choice(pool, size=train_size, replace=False)
  pool = np.delete(pool, index_train)
  index_val = np.random.choice(pool, size=val_size, replace=False)
  index_train = np.sort(index_train)
  index_val = np.sort(index_val)
  return index_train, index_val
  


#load the datasets and split accordingly to the global definition
#we split this functionality to gurantee manual augmentation of data on same randomly-split datasets
def get_datasets(path, train_size, val_size):
  #H array containing both source and target images
  ds = Load_HDF5(path, get_all=True, transform=ToTensor())
  tr_idx, val_idx = sample_index(train_size, val_size, train_size+val_size)
  tr_idx = tr_idx.tolist()
  val_idx = val_idx.tolist()
  #split data into train, validation, test sets
  tr_set = Subset(ds, tr_idx)
  val_set = Subset(ds, val_idx)
  return  tr_set, val_set



#given train, valid, and test sets, augment data 
def aug_datasets(dataset, split_proportion, batch_size, grid_size, augment_times, mask_size, val_set=False):
  data_size = len(dataset)
  tar, src = random_split(dataset, [int(data_size*split_proportion), data_size - int(data_size*split_proportion)])
  tar_aug = Augment(tar, batch_size, 128, augment_times = augment_times, transform = True, random_mask = True, transform_no =2, val_set=val_set)
  src_aug = Expand(src, len(tar_aug))
  return tar_aug, src_aug

#given dataset, return the appropriate dataloader
def get_dataloader(ds, batch_size, augment=False, shuffle=False):
  SHUFFLE = shuffle 
  if augment:
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn, shuffle=SHUFFLE)
  else:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=SHUFFLE)  
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


def get_val_dl(val, split_proportion, batch_size, grid_size, augment_times, mask_size):
  tar, src = aug_datasets(val, split_proportion, batch_size, grid_size, augment_times, mask_size, val_set=True)
  val_tar_dl = get_dataloader(tar, batch_size, augment=True, shuffle=True)
  val_src_dl = get_dataloader(src, batch_size, shuffle=True)
  #pre augment the data by loading them before execution
  val_tar_ds  = pre_augment(val_tar_dl, batch_size=batch_size, val_set=True, shuffle=True)
  val_tar_dl = DataLoader(val_tar_ds, batch_size, shuffle=True)
  return val_tar_dl, val_src_dl
  
