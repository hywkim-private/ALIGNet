import config
import torch 
import model

def save_model(checkpoint, checkpoint_path):
  path = checkpoint_path + '/checkpoint.pt'
  torch.save(checkpoint, path)
  


#save a dictionary of datasets
def save_ds(ds, ds_path):
  path = ds_path / 'ds.pt'
  torch.save(ds, path)


#load the model
def load_model(path):
  model = ALIGNet(GRID_SIZE)
  model_l = torch.load(path)
  model.load_state_dict(model_l)
  return model

