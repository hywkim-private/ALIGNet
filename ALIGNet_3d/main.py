import argparse
import os
import torch 
import numpy as np
import run_3d, config_3d, model, preprocess, test, validate_3d
from model import io_3d, ops_3d, network_3d
from preprocess import datasets
import config_parse 


#TODO: UPDATE  THE DATATYPE CHECKING CAPABIILITY USING DATATYPE AND DATA IDX IN INIT_CONFIG
#FIRST ENABLE CONFIG FOR EACH DATA FILES
#THEN UPDATE THE CORRESPODING ERROR CHECKING CAPABILITIES


#TODO: MAKE THE RENDER VALID CONFIG FUNCTION AND TEST
#MAKE SURE WE ARE NOT USING COMMAND LINE ARGUMENTS ANYMMORE BUT ENTIRELY DEPENDENT ON CONFIG FILES


#TODO: CREATE COLLATE_FN FOR DS IN CONVERT_TYPE.PY


#MAJOR ERROR: EITHER SRC BATCH IS BEING FLIPPED SOMEWHERRE OR GRID_SAMPLE NOT TAKING FROM THE RIGHT VOXELS

#given a string of datatype, return its appropriate index
def get_data_idx(datatype_str):
  dtype = 0
  if datatype_str == "plane":
    dtype = 0
  return dtype
    

#load and augment train and valid datasets
def get_ds(file_path, tr_size, val_size, total_num, pt_sample):
  #first check if the requested tr_size is not larger than the number of files present
  num_files = len(os.listdir(file_path))
  if tr_size > num_files:
    print(f"The requested train size {tr_size} is greater than the number of files present: {num_files} in directory {file_path}")
    exit()
  if val_size > num_files:
    print(f"The requested train size {tr_size} is greater than the number of files present: {num_files} in directory {file_path}")
    exit()
  tr_index, val_index = datasets.sample_index(tr_size, val_size, total_num)
  tr_mesh = io_3d.Load_Mesh(file_path, sample_index=tr_index)
  val_mesh = io_3d.Load_Mesh(file_path, sample_index=val_index)
  #here we convert mesh datatype into pointlcoud and voxel datatype
  tr, val = datasets.get_datasets_3d(tr_mesh,val_mesh, config_3d.VOX_SIZE, pt_sample)
  return tr, val
  
#common code snippet--run the train iteration loop
def train_model(tr, val, model_path):
  #save the iter variable as default or specified by the user
  iter_t = config_3d.ITER
  #make an overfit check if specified by config 
  result_check = None 
  result_check_path = config_3d.MODEL_PATH + 'result_checker/'
  if config_3d.RESULT_CHECK:
    #make the valid dataset
    val_tar, val_src = preprocess.datasets.get_val_dl_3d(
      val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, config_3d.MASK_SIZE)
    if os.path.exists(result_check_path): 
      result_check = io_3d.load_obj(result_check_path + 'result_checker.obj')
    else:
      os.makedirs(result_check_path)
      result_check = validate_3d.result_checker_3d(model, val_tar, val_src)
  run_3d.train_3d(model, model_path, tr, result_checker=result_check, graph_loss=config_3d.GRAPH_LOSS)
  #model is already saved in the run operation
  if config_3d.RESULT_CHECK:
    io_3d.save_obj(result_check, result_check_path)

if __name__ == '__main__':
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
  new.add_argument("-n", "--name",  type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type",  type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10") 
  new.add_argument("-m", "--model", type=int, help="the type index of the model on which to train ALIGNet. default: 1")
  new.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")
  #download and save new dataset
  data.add_argument("-ty", "--type",  type=str,  help="Specify the dataset to use for the new model")
  data.add_argument("-d", "--download", action='store_true', help ="Download the data and get the dataset")
  #train a model previously created
  train.add_argument("-ty", "--type",  type=str,  help="Specify the dataset to use for the new model")
  train.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10")
  train.add_argument("-n", "--name", type=str,  help="Specify the name of the model")
  train.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")

  #validate the model using the valid dataset
  valid.add_argument("-ty", "--type", type=str, help="Specify the dataset to use for the new model")
  valid.add_argument("-n", "--name",  type=str,  help="Specify the name of the model")
  valid.add_argument("-v", "--visualize", type=str, help='Specify whether to visualize, and provide the dataformat: pointcould, voxel, or mesh')
  valid.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")

  args = ap.parse_args()
  #load the configuartion file
  config = config_parse.load_config('init_config.yaml')

  

  #if new data, download data, create, and save
  if args.command == 'data': 
    #load config
    config = config_parse.load_config('./init_config.yaml')
    config_parse.render_data_config(config)
    #check the type of data (for now, we will only support plane--add more)
    #first, check for existing directory and if not, create a new directory to store data 
    #we assume a all-inclusive imagenet dataset, so only need to download once 
    url = config_3d.URL_DATA
      
    #download data
    if not os.path.exists(config_3d.DATA_PATH):
      os.makedirs(config_3d.DATA_PATH)
      
    #write configuration file
    config_parse.write_data_config(config, config_3d.DATA_PATH)
    #download only if the download flag is set
    if args.download:
      model.io_3d.download_zip(config_3d.URL_DATA, config_3d.DATA_PATH)
    dtype = get_data_idx(config_3d.DATA_TYPE)
    tr, val = get_ds(config_3d.LOAD_DATA_PATH, config_3d.TRAIN_SIZE, config_3d.VAL_SIZE, config_3d.TRAIN_SIZE+config_3d.VAL_SIZE, config_3d.PT_SAMPLE)
    
    #save datasets to the same directory as the model
    model.io_3d.save_ds(tr, 'tr', config_3d.DATA_PATH)
    model.io_3d.save_ds(val, 'val', config_3d.DATA_PATH)
    

  #if new model, download and create dataset, make new model
  if args.command == 'new':
    #make new  model
    #----------------handle config---------------------
    #load and save the model configuration
    config = config_parse.load_config('./init_config.yaml')
    #laod train configuration from the init_config file
    config_parse.render_train_config(config)
    #from the train config, find the model path and load its config
    config_parse.render_model_config(config)
    #from the model config, find the data configurations and load its config 
    data_config = config_parse.load_config(config_3d.DATA_PATH + 'data_cfg.yaml') 
    config_parse.render_data_config(data_config)
    #----------------------------------------------------
    #first check if the dataset exists
    if not os.path.exists(config_3d.DATA_PATH):
      print(f"Dataset not found in path {config_3d.DATA_PATH}: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if os.path.exists(config_3d.MODEL_PATH):
      print(f"Model already exists in {config_3d.MODEL_PATH}: use [train] argument to do additional training or create a model with different name")
      exit()
    #make a directory to save model and dataset
    if not os.path.exists(config_3d.MODEL_PATH):
        os.makedirs(config_3d.MODEL_PATH)
        os.makedirs(config_3d.MODEL_PATH + 'outputs/')
        os.makedirs(config_3d.MODEL_PATH + 'outputs/loss_graphs/')
        os.makedirs(config_3d.MODEL_PATH + 'outputs/images/')
        #write model configuration 
        config_parse.write_model_config(config, config_3d.MODEL_PATH)
        
    init_grid = ops_3d.init_grid_3d(config_3d.GRID_SIZE).view(-1).to(config_3d.DEVICE)
    model = network_3d.ALIGNet_3d(config_3d.MODEL_IDX, config_3d.GRID_SIZE, config_3d.VOX_SIZE, init_grid, config_3d.MAXFEAT, learn_offset=config_3d.LEARN_OFFSET).to(config_3d.DEVICE)
    #load data and its configuarations
    tr, val = io_3d.load_ds(config_3d.DATA_PATH )
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    train_model(tr, val, config_3d.MODEL_PATH)
    
      
  if args.command == 'train':
    #load the target model configuartions 
    #--------------------------handle config----------------------
    #render the model configurations
    config_parse.render_train_config(config)
    model_config = config_parse.load_config(config_3d.MODEL_PATH + 'model_cfg.yaml')
    config_parse.render_model_config(model_config)
    data_config = config_parse.load_config(config_3d.DATA_PATH + 'data_cfg.yaml')
    config_parse.render_data_config(data_config)
    #--------------------------------------------------------------
    #we will check the initialized directories to verify that models/datasets are initialized
    #the commands below (valid and train) both need to be checked if the model exists or not -- error if  not 
    if not os.path.exists(config_3d.DATA_PATH):
      print(f"Dataset not found in path {config_3d.DATA_PATH}: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if not os.path.exists(config_3d.MODEL_PATH):
      print(f"Model not found in path {config_3d.MODEL_PATH}: create model using [new] argument")
      exit()
    model = torch.load(config_3d.MODEL_PATH + 'model.pt', map_location=torch.device('cpu')).to(config_3d.DEVICE)
    model = model.to(config_3d.DEVICE)
    tr, val = io_3d.load_ds(config_3d.DATA_PATH)
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    train_model(tr, val, config_3d.MODEL_PATH)
    
  if args.command == 'valid':
    #--------------------------handle config----------------------
    #render valid configurations from init_config 
    config_parse.render_valid_config(config)
    #render model configurations from model_cfg.yaml found in the model path
    model_config = config_parse.load_config(config_3d.MODEL_PATH + 'model_cfg.yaml')
    config_parse.render_model_config(model_config)
    data_config = config_parse.load_config(config_3d.DATA_PATH + 'data_cfg.yaml')
    config_parse.render_data_config(data_config)
    #--------------------------------------------------------------
    
    #load the previous model
    model = torch.load(config_3d.MODEL_PATH + 'model.pt', map_location=torch.device('cpu')).to(config_3d.DEVICE)
    #load the target model configuartions 
    model_config = config_parse.load_config(config_3d.MODEL_PATH + 'model_cfg.yaml')
    #path to save the visualized image 
    image_path = config_3d.MODEL_PATH + 'outputs/images/'
    image_path = io_3d.latest_filename_png(image_path)
    #load the neccessary datasets
    tr, val = io_3d.load_ds(config_3d.DATA_PATH)
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    #get the target and val sets
    #visualize if specified
    #pointcloud
    #DEPRECATED
    if config_3d.VISUALIZE_TYPE == 'pointcloud':
      print("This function is deprecated and is no longer supported => aborting..")
      exit()
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, config_3d.MASK_SIZE, val_sample= config_3d.NUM_SAMPLE, )
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      result_checker.update()
      result_checker.get_pointcloud_from_mesh(config_3d.PT_SAMPLE)
      result_checker.visualize(datatype=2, save_path=image_path)
      
    #visualize mesh
    #DEPRECATED
    elif config_3d.VISUALIZE_TYPE == 'mesh':
      print("This function is deprecated and is no longer supported => aborting..")
      exit()
      #for mesh visualization, we need to set get_mesh=True
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, config_3d.MASK_SIZE, val_sample= config_3d.NUM_SAMPLE, get_src_mesh=True, get_tar_pt=False)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      #we will retrieve the original src mesh representation and apply deformation directly
      result_checker.update(get_mesh=True)
      result_checker.warp_mesh()
      result_checker.get_mesh_from_vox()
      result_checker.visualize(datatype=1, save_path=image_path)
      
    #a custom visualization routine designed for research purposes
    elif config_3d.VISUALIZE_TYPE == 'custom':
      #for mesh visualization, we need to set get_mesh=True, get_for_grid=True
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, config_3d.MASK_SIZE, val_sample= config_3d.NUM_SAMPLE, get_src_mesh=True, get_tar_pt=True)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      #we will retrieve the original src mesh representation and apply deformation directly
      result_checker.update(get_src_mesh=True, get_tar_pt = True, get_for_grid=True)
      result_checker.warp_mesh()
      result_checker.visualize(datatype=1, save_path=image_path)
    else:
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, config_3d.MASK_SIZE)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      result_checker.update()
      result_checker.visualize(datatype=0, save_path=image_path)
 
    
      
      
      
    
    
