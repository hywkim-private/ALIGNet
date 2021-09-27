import config
import model
import validate
import train
import load_data
import load_save
import argparse
import os
import torch 



if __name__ == '__main__':
  
  def train():
    #decide and load which type of data to use
    if args.type == 'vase':
      d_type = config.VASE 
    elif args.type == 'plane ':
      d_type = config.PLANE
    #load datasets
    tr, val, test = load_ds(config.FILE_PATH + args.type + '/', d_type)
    #save the iter variable as default or specified by the user
    iter = 10
    if args.iter: 
      iter = args.iter
    #make an overfit check if specified by config 
    overfit_check = None 
    if config.OVERFIT_CHECK:
       #make the valid dataset
      val_tar, val_src, val_tar_aug, val_src_aug = load_data.get_val_dl(tr, val, test)
      overfit_check = validate.overfit_checker(model, val_tar, val_src)
    train.train(model, iter, tr, val, test, args.name, overfit_checker=overfit_check, train_mode=config.TRAIN_MODE, graph_loss=config.GRAPH_LOSS)
    torch.save(model.state_dict(), config.MODEL_PATH + args.name +'/'+ args.name+'.pt')
    
  config.initialize_config()
  #argument parser
  ap = argparse.ArgumentParser()
  subap = ap.add_subparsers(dest='command')
  new = subap.add_parser('new')
  train = subap.add_parser('train')
  val = subap.add_parser('val')
  data = subap.add_parser('data')

  #if -n argument is given, create a new model
  new.add_argument("-f", "--filepath", type=str,  help="Specify the filepath to create newmodel")
  new.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=str, help="number of times to run the training loop, default: 10") 
  
  #download and save new dataset
  data.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  
  #train a model previously created
  train.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  train.add_argument("-i", "--iter", type=str, help="number of times to run the training loop, default: 10")
  train.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the model")
  
  args = ap.parse_args()
  #if new data, download data, create, and save
  if args.command == 'data': 
    #first, check for existing directory and if not, create a new directory to store data 
    if args.type == 'vase':
      file_path = config.FILE_PATH_VASE
      d_type = config.VASE
      url = config.URL_VASE
      
    if args.type == 'plane':
      file_path = config.FILE_PATH_PLANE
      d_type = config.PLANE
      url = config.URL_PLANE
      
    #download data
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path +'/datasets')
        
    load_data.download_data(url, d_type)
    tr, val, test = load_data.get_datasets(d_type)
    
    #save datasets to the same directory as the model
    load_save.save_ds(tr, 'tr', file_path +'/datasets/')
    load_save.save_ds(val, 'val', file_path +'/datasets/')
    load_save.save_ds(test, 'test', file_path + '/datasets/')
    
    
    
  #if new model, download and create dataset, make new model
  if args.command == 'new':
    #first check if the dataset exists
    if not os.path.exists(config.FILE_PATH + args.type + '/'):
      print("Dataset not found: create dataset using [data] argument")
      exit()
    save_path = config.FILE_PATH + args.name
    #make new  model
    model = model.ALIGNet(config.GRID_SIZE)
    torch.save(model.state_dict(), config.MODEL_PATH + args.name +'/'+ args.name+'.pt')
    #make a directory to save model and dataset
    if not os.path.exists(config.MODEL_PATH + args.name):
        os.makedirs(config.MODEL_PATH + args.name)
    train()
    
    
  if args.command == 'train':
    #first check if the dataset exists
    if not os.path.exists(config.FILE_PATH + args.type + '/'):
      print("Dataset not found: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if not os.path.exists(config.MODEL_PATH + args.name + '/'):
      print("Model not found: create model using [new] argument")
      exit()
    #load the previous model
    model_path = './' + args.name + '/' + args.name + '.pt'
    model = load_model(model_path)
    train()
    
      
      
      
    
    
  