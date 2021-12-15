import config
import model
import validate
from utils import etc, load_save
import train as train_
import load_data
import argparse
import os
import torch 
import config_parse

#given a string of datatype, return its appropriate index
def get_data_idx(datatype_str):
  dtype = 0
  if datatype_str == "plane":
    dtype = 0
  return dtype
    

    
def train_model(tr, val, model):
  #make an overfit check if specified by config 
  result_check = None 
  if config.RESULT_CHECK:
    result_check_path = config.MODEL_PATH + '/result_checker.obj'
    #make the valid dataset
    val_tar, val_src  = load_data.get_val_dl(
      val, config.TARGET_PROPORTION_VAL, config.BATCH_SIZE, config.GRID_SIZE, config.AUGMENT_TIMES_VAL, config.MASK_SIZE)
    if os.path.exists(result_check_path): 
      result_check = load_obj(result_check_path)
    else:
      result_check = validate.result_checker(model, val_tar, val_src)
  train_.train(model, config.ITER, tr, config.TARGET_PROPORTION, config.BATCH_SIZE, config.GRID_SIZE, config.AUGMENT_TIMES_TR, config.MASK_SIZE, result_checker=result_check, graph_loss=config.GRAPH_LOSS)
  torch.save(model, config.MODEL_PATH + 'model.pt')
  if config.RESULT_CHECK:
    load_save.save_obj(result_check, result_check_path)



if __name__ == '__main__':
  config.initialize_config()
  #argument parser
  ap = argparse.ArgumentParser()
  subap = ap.add_subparsers(dest='command')
  new = subap.add_parser('new')
  train = subap.add_parser('train')
  valid = subap.add_parser('valid')
  data = subap.add_parser('data')

  #if -n argument is given, create a new model
  new.add_argument("-f", "--filepath", type=str,  help="Specify the filepath to create newmodel")
  new.add_argument("-n", "--name",  type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type", type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10") 
  
  #download and save new dataset
  data.add_argument("-ty", "--type",  type=str,  help="Specify the dataset to use for the new model")
  data.add_argument("-d", "--download", action='store_true', help ="Download the data and get the dataset")

  #train a model previously created
  train.add_argument("-ty", "--type",  type=str,  help="Specify the dataset to use for the new model")
  train.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10")
  train.add_argument("-n", "--name", type=str,  help="Specify the name of the model")
  
  #validate the model using the valid dataset
  valid.add_argument("-ty", "--type",  type=str, help="Specify the dataset to use for the new model")
  valid.add_argument("-n", "--name", type=str,  help="Specify the name of the model")
  valid.add_argument("-v", "--visualize", type=str, help='Specify whether or not to visualize the results')
  
  args = ap.parse_args()
  #load the configuartion file
  cfg = config_parse.load_config('init_config.yaml')


  #if new data, download data, create, and save
  if args.command == 'data': 
    #load config
    config_parse.render_data_config(cfg)
    #check the type of data (for now, we will only support plane--add more)
    #first, check for existing directory and if not, create a new directory to store data 
    #we assume a all-inclusive imagenet dataset, so only need to download once 
    url = config.URL_DATA
      
    #download data
    if not os.path.exists(config.DATA_PATH):
      os.makedirs(config.DATA_PATH)
    #write configuration file
    config_parse.write_data_config(cfg, config.DATA_PATH)
    dtype = get_data_idx(config.DATA_TYPE)
    #download the data if download argument is specified
    if args.download:
      if not os.path.exists(config.LOAD_DATA_PATH):
        os.makedirs(config.LOAD_DATA_PATH)
      #download data and save it to the datapath
      load_data.download_data(config.URL_DATA, config.LOAD_DATA_PATH)
      tr, val = load_data.get_datasets(config.LOAD_DATA_PATH, config.TRAIN_SIZE, config.VAL_SIZE)
      #save datasets to the same directory as the model
      load_save.save_ds(tr, 'tr', config.DATA_PATH)
      load_save.save_ds(val, 'val', config.DATA_PATH)
    else:
      tr, val = load_data.get_datasets(config.LOAD_DATA_PATH, config.TRAIN_SIZE, config.VAL_SIZE)
      #save datasets to the same directory as the model
      load_save.save_ds(tr, 'tr', config.DATA_PATH)
      load_save.save_ds(val, 'val', config.DATA_PATH)
    


  #if new model, download and create dataset, make new model
  if args.command == 'new':
    #make new  model
    #----------------handle config---------------------
    #load and save the model configuration
    cfg = config_parse.load_config('./init_config.yaml')
    #laod train configuration from the init_config file
    config_parse.render_train_config(cfg)
    #from the train config, find the model path and load its config
    config_parse.render_model_config(cfg)
    #from the model config, find the data configurations and load its config 
    data_config = config_parse.load_config(config.DATA_PATH + 'data_cfg.yaml') 
    config_parse.render_data_config(data_config)
    #----------------------------------------------------
    #first check if the dataset exists
    if not os.path.exists(config.DATA_PATH):
      print(f"Dataset not found in path {config.DATA_PATH}: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if os.path.exists(config.MODEL_PATH):
      print(f"Model already exists in {config.MODEL_PATH}: use [train] argument to do additional training or create a model with different name")
      exit()
    #make a directory to save model and dataset
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
        os.makedirs(config.MODEL_PATH + 'outputs/')
        os.makedirs(config.MODEL_PATH + 'outputs/loss_graphs/')
        os.makedirs(config.MODEL_PATH + 'outputs/images/')
        #write model configuration 
        config_parse.write_model_config(cfg, config.MODEL_PATH)
    model = model.ALIGNet(config.GRID_SIZE).to(config.DEVICE)
    tr, val = load_data.load_ds(config.DATA_PATH)
    train_model(tr, val, model)
    
    

  if args.command == 'train':
    #load the target model configuartions 
    #--------------------------handle config----------------------
    #render the model configurations
    config_parse.render_train_config(cfg)
    model_config = config_parse.load_config(config.MODEL_PATH + 'model_cfg.yaml')
    config_parse.render_model_config(model_config)
    data_config = config_parse.load_config(config.DATA_PATH + 'data_cfg.yaml')
    config_parse.render_data_config(data_config)
    #--------------------------------------------------------------
    #we will check the initialized directories to verify that models/datasets are initialized
    #the commands below (valid and train) both need to be checked if the model exists or not -- error if  not 
    if not os.path.exists(config.DATA_PATH):
      print(f"Dataset not found in path {config.DATA_PATH}: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if not os.path.exists(config.MODEL_PATH):
      print(f"Model not found in path {config.MODEL_PATH}: create model using [new] argument")
      exit()
    model = torch.load(config.MODEL_PATH + 'model.pt', map_location=torch.device('cpu')).to(config.DEVICE)
    tr, val = load_data.load_ds(config.DATA_PATH)
    train_model(tr, val, model)
    
  if args.command == 'valid':
    #--------------------------handle config----------------------
    #render valid configurations from init_config 
    config_parse.render_valid_config(cfg)
    #render model configurations from model_cfg.yaml found in the model path
    model_config = config_parse.load_config(config.MODEL_PATH + 'model_cfg.yaml')
    config_parse.render_model_config(model_config)
    data_config = config_parse.load_config(config.DATA_PATH + 'data_cfg.yaml')
    config_parse.render_data_config(data_config)
    #--------------------------------------------------------------
    
    #load the previous model
    model = torch.load(config.MODEL_PATH + 'model.pt', map_location=torch.device('cpu')).to(config.DEVICE)
    #load the target model configuartions 
    model_config = config_parse.load_config(config.MODEL_PATH + 'model_cfg.yaml')
    #path to save the visualized image 
    image_path = config.MODEL_PATH + 'outputs/images/'
    image_path = etc.latest_filename(image_path)
    #load the neccessary datasets
    tr, val = load_data.load_ds(config.DATA_PATH)
    val_tar_dl, val_src_dl = load_data.get_val_dl(
      val, config.TARGET_PROPORTION_VAL, config.BATCH_SIZE, config.GRID_SIZE, config.AUGMENT_TIMES_VAL, config.MASK_SIZE)
    result_checker = validate.result_checker(model, val_tar_dl, val_src_dl, checker_board=True)
    result_checker.update()
    result_checker.visualize(image_path)
      
      
      
      