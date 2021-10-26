import torch 
import pickle 
import os
from pathlib import Path
from pytorch3d import io
from zipfile import ZipFile
from model import network_3d
import wget

#a function that parses through the file and returns the filename of the highest index number
def latest_filename(file_path):
  file_list = []
  for file in glob.glob(file_path + '/*.png', recursive=True):
    str_list = str(file).split('/')
    file = str_list[len(str_list)-1]
    file = file[0]
    file = int(file)
    file_list.append(file)
  if not len(file_list) == 0:
    max_num = max(file_list) + 1
  else:
    max_num = 0
  filename = file_path  + str(max_num) + '.png'
  return filename
    

#download and extract zip_file data
def download_zip(url, extract_path):
  filename = wget.download(url)
  if os.path.exists(result_check_path): 
    obj = preprocess.io_3d.load_obj(result_check_path)
  zf = ZipFile(filename, 'r')
  zf.extractall(extract_path)
  zf.close()

#read mesh format data
#the sample_index is a list of all indexes to sample from the given file directory 
#total data specifies how many data are in the file
#returns: pytorch mesh object
def Load_Mesh(file_path, sample_index = None):
  p = Path(file_path)
  files = sorted(p.glob('*.obj'))
  if sample_index.any:
    data_list = []
    for i in sample_index:
      file_ = files[i]
      data_list.append(file_)
    data = io.load_objs_as_meshes(data_list)
  else:
    data = io.load_objs_as_meshes(files)
  return data
  

def save_model(checkpoint, checkpoint_path, filename):
  path = checkpoint_path + filename
  torch.save(checkpoint, path)
  


#save a dictionary of datasets
def save_ds(ds, ds_name, ds_path):
  path = ds_path + ds_name + '.pt'
  torch.save(ds, path)
  
#load raw datasets from dir
def load_ds(path, ds_index=0):
  #TODO: add more datasets
  ds_name = 'plane'
  #if ds_index == 1:
    #ds_name = 'plane'
  tr = torch.load(os.path.join(path, 'tr.pt'))
  val = torch.load(os.path.join(path, 'val.pt'))
  return tr, val

#load the model
def load_model(path, name, grid_size):
  model_ = network_3d.ALIGNet_3d(name, grid_size)
  model_l = torch.load(path)
  model_.load_state_dict(model_l)
  return model_


#a function to save objects--namely the result_checker
def save_obj(obj, save_path):
  file_pi = open(save_path + '.obj', 'wb') 
  pickle.dump(obj, file_pi)

#a function to load objects
def load_obj(load_path):
  filehandler = open(load_path, 'rb') 
  obj = pickle.load(filehandler)
  return obj

