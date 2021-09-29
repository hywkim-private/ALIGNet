import config
import torch 
import model
import pickle 

def save_model(checkpoint, checkpoint_path):
  path = checkpoint_path + '/checkpoint.pt'
  torch.save(checkpoint, path)
  


#save a dictionary of datasets
def save_ds(ds, ds_name, ds_path):
  path = ds_path + ds_name + '.pt'
  torch.save(ds, path)


#load the model
def load_model(path, name):
  model_ = model.ALIGNet(name, config.GRID_SIZE)
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