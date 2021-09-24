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


#load raw plane datasets
def load_ds(path, ds_index=0):
  ds_name = 'vase'
  if ds_index == 1:
    ds_name = 'plane'

  tr = torch.load(path+'trainset_' + ds_name + '.pt')
  val = torch.load(path+'valset_' + ds_name + '.pt')
  test = torch.load(path+'testset_' + ds_name + '.pt')
  return tr, val, test
