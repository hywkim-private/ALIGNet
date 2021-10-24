import argparse
import os
import torch 
from model import io_3d

import run_3d, config_3d, model, preprocess, test
from model import io_3d, ops_3d, network_3d

if __name__ == '__main__':
  #common code snippet--load data as specified
  def load_ds():
    #decide and load which type of data to use (ADD MORE DATA)
    if args.type == 'plane ':
      d_type = config.PLANE
    #load datasets
    tr, val, test = io_3d.load_ds(os.path.join(config_3d.MODEL_PATH,args.name, args.type,'/datasets/'), d_type)
    return tr, val, test
    
  #common code snippet--run the train iteration loop
  def train_model():
    #save the iter variable as default or specified by the user
    iter = 10
    if args.iter: 
      iter = args.iter
    #make an overfit check if specified by config 
    result_check = None 
    result_check_path = config.MODEL_PATH + args.name + '/result_checker'
    if config_3d.RESULT_CHECK:
       #make the valid dataset
      val_tar, val_src, val_tar_aug, val_src_aug = preprocess.datasets.get_val_dl(tr, val)
      if os.path.exists(result_check_path): 
        obj = preprocess.io_3d.load_obj(result_check_path)
      else:
        result_check = test.validate_3d.result_checker_3d(model, val_tar, val_src)
    run_3d.train_3d(model, iter, tr, val, test, args.name, result_checker=result_check, train_mode=config.TRAIN_MODE, graph_loss=config.GRAPH_LOSS)
    torch.save(model.state_dict(), config_3d.MODEL_PATH + args.name +'/'+ args.name+'.pt')
    preprocess.io_3d.save_obj(result_check, result_check_path)
  
  #initialize all config settings
  config_3d.initialize_config()
  #argument parser
  ap = argparse.ArgumentParser()
  subap = ap.add_subparsers(dest='command')
  new = subap.add_parser('new')
  train = subap.add_parser('train')
  valid = subap.add_parser('valid')
  data = subap.add_parser('data')

  #if -n argument is given, create a new model
  new.add_argument("-f", "--filepath", type=str,  help="Specify the filepath to create newmodel")
  new.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10") 
  
  #download and save new dataset
  data.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  
  #train a model previously created
  train.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  train.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10")
  train.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the model")
  
  #validate the model using the valid dataset
  valid.add_argument("-ty", "--type", required=True, type=str, help="Specify the dataset to use for the new model")
  valid.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the model")
  valid.add_argument("-v", "--visualize", type=str, help='Specify whether to visualize, and provide the dataformat: pointcould, voxel, or mesh')
  
  
  args = ap.parse_args()
  #if new data, download data, create, and save
  if args.command == 'data': 
    #first, check for existing directory and if not, create a new directory to store data 
    #we assume a all-inclusive imagenet dataset, so only need to download once  
    path = config_3d.DATA_PATH
    url = config_3d.URL_DATA
      
    #download data
    if not os.path.exists(path):
        os.makedirs(path)
      
        
    model.io_3d.download_zip(config_3d.URL_DATA, config_3d.DATA_PATH)
    tr, val, test = model.io_3d.get_datasets(d_type)
    
    #save datasets to the same directory as the model
    model.io_3d.save_ds(tr, 'tr', path )
    model.io_3d.save_ds(val, 'val', path )
    model.io_3d.save_ds(test, 'test', path )
    

  #check the type of data (for now, we will only support plane--add more)
  if args.type == 'plane':
    data_path = config_3d.DATA_PATH_PLANE
    data_index = 0
  #get the path to the target model
  model_path = config_3d.MODEL_PATH + args.name + '/'
  model_obj_path = model_path + args.name + '.pt'

    
  #if new model, download and create dataset, make new model
  if args.command == 'new':
    #first check if the dataset exists
    if not os.path.exists(data_path):
      print(f"Dataset not found in path {data_path}: create dataset using [data] argument")
      exit()
    save_path = config_3d.DATA_PATH + args.name
    #make new  model
    init_grid = ops_3d.init_grid_3d(config_3d.GRID_SIZE).view(-1).to(config_3d.DEVICE)
    model = network_3d.ALIGNet_3d(args.name, config_3d.GRID_SIZE, config_3d.VOX_SIZE, init_grid).to(config_3d.DEVICE)
    
    #make a directory to save model and dataset
    if not os.path.exists(config_3d.MODEL_PATH + args.name):
        os.makedirs(config_3d.MODEL_PATH + args.name)
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs')
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs/loss_graphs')
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs/images')
    tr, val, test = load_ds()
    train_model()
    
    
  #we will check the initialized directories to verify that models/datasets are initialized
  #the commands below (valid and train) both need to be checked if the model exists or not -- error if  not 
  if not os.path.exists(data_path):
    print(f"Dataset not found in path {data_path}: create dataset using [data] argument")
    exit()
  #check if the dataset exists
  if not os.path.exists(model_path):
    print(f"Model not found in path {model_path}: create model using [new] argument")
    exit()
    
  if args.command == 'train':
    #load the target model
    model_obj_path = model_path + args.name + '.pt'
    model = preprocess.io_3d.load_model(model_obj_path, args.name, config_3d.GRID_SIZE).to(config_3d.DEVICE)
    model = model.to(config_3d.DEVICE)
    tr, val, test = preprocessio_3d.load_ds(data_path, data_index)
    train_model()
    
  
  if args.command == 'valid':
    #load the previous model
    model = preprocess.io_3d.load_model(model_obj_path, args.name,config_3d.GRID_SIZE).to(config_3d.DEVICE)
    model= model.to(config_3d.DEVICE)
    #path to save the visualized image 
    image_path = model_path + '/outputs/images/'
    image_path = preprocess.io_3d.latest_filename(image_path)
    #load the neccessary datasets
    tr, val, test = preprocess.io_3d.load_ds(data_path, data_index)
    val_tar_dl, val_src_dl, val_tar_dl_aug, val_src_dl_aug = preprocess.datasets.get_val_dl_3d(tr, val, test)
    result_checker = test.validate_3d.result_checker_3d(model_plane, val_tar_dl, val_src_dl)
    result_checker.update()
    #visualize if specified
    #pointcloud
    if args.visualize == 'pointcloud':
      result_checker2.get_pointcloud(config_3d.PT_SAMPLE)
      result_checker2.visualize(config_3d.NUM_SAMPLE, datatype=1)
    #mesh--NOT WORKING (RETURN ERROR FOR NOW)
    elif args.visualize == 'mesh':
      print("ERROR: MESH VISUALIZATION IS CURRENTLY UNDER DEVELOPMENT--USE A DIFFERENT FORMAT")
    #voxel
    else:
      result_checker2.visualize(config_3d.NUM_SAMPLE, datatype=0)
 
    
      
      
      
    
    
  