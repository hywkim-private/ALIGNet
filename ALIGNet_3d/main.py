import argparse
import os
import torch 
import numpy as np
from model import io_3d
import run_3d, config_3d, model, preprocess, test, validate_3d
from model import io_3d, ops_3d, network_3d
from preprocess import datasets

#given a string of datatype, return its appropriate index
def get_data_idx(datatype_str):
  dtype = 0
  if datatype_str == "plane":
    dtype = 0
  return dtype
    
#common code snippet--load data as specified
#input
#ds_idx: the dataset index 
def load_ds(ds_idx):
  #decide and load which type of data to use (ADD MORE DATA)
  #default d_type = plane
  dtype = get_data_idx(args.type)
  #load datasets
  tr, val = io_3d.load_ds(config_3d.DATA_PATH +'datasets/'+args.type+'/', ds_type=dtype, ds_idx=ds_idx)
  return tr, val
  
#load and augment train and valid datasets
def get_ds(file_path, tr_size, val_size, total_num, pt_sample):
  tr_index, val_index = datasets.sample_index(tr_size, val_size, total_num)
  tr_mesh = io_3d.Load_Mesh(file_path, sample_index=tr_index)
  val_mesh = io_3d.Load_Mesh(file_path, sample_index=val_index)
  #here we convert mesh datatype into pointlcoud and voxel datatype
  tr, val = datasets.get_datasets_3d(tr_mesh,val_mesh, config_3d.VOX_SIZE, pt_sample)
  return tr, val
  
#common code snippet--run the train iteration loop
def train_model(args, tr, val, model_path):
  #save the iter variable as default or specified by the user
  iter_t = 10
  if args.iter: 
    iter_t = args.iter
  #make an overfit check if specified by config 
  result_check = None 
  result_check_path = config_3d.MODEL_PATH + args.name + '/result_checker'
  if config_3d.RESULT_CHECK:
    #make the valid dataset
    val_tar, val_src = preprocess.datasets.get_val_dl_3d(
      val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL)
    if os.path.exists(result_check_path): 
      obj = preprocess.io_3d.load_obj(result_check_path)
    else:
      result_check = validate_3d.result_checker_3d(model, val_tar, val_src)
  run_3d.train_3d(model, model_path, iter_t, tr, args.name, result_checker=result_check, graph_loss=config_3d.GRAPH_LOSS)
  #model is already saved in the run operation
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
  new.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10") 
  new.add_argument("-m", "--model", type=int, help="the type index of the model on which to train ALIGNet. default: 1")
  new.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")
  #download and save new dataset
  data.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  data.add_argument("-d", "--download", action='store_true', help ="Download the data and get the dataset")
  #train a model previously created
  train.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  train.add_argument("-i", "--iter", type=int, help="number of times to run the training loop, default: 10")
  train.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the model")
  train.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")

  #validate the model using the valid dataset
  valid.add_argument("-ty", "--type", required=True, type=str, help="Specify the dataset to use for the new model")
  valid.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the model")
  valid.add_argument("-v", "--visualize", type=str, help='Specify whether to visualize, and provide the dataformat: pointcould, voxel, or mesh')
  valid.add_argument("-d", "--data", type=int, help="the index of the dataset=> default:0")

  args = ap.parse_args()
  
   
  #check the type of data (for now, we will only support plane--add more)
  if args.type == 'plane':
    data_path = config_3d.DATA_PATH_PLANE
    data_index = 0

  #if new data, download data, create, and save
  if args.command == 'data': 
    #first, check for existing directory and if not, create a new directory to store data 
    #we assume a all-inclusive imagenet dataset, so only need to download once 
    #TODO: add other datatypes as if statement
    load_path = config_3d.DATA_PATH_PLANE
    save_path = config_3d.DATA_PATH + 'datasets/' + args.type + '/'
    url = config_3d.URL_DATA
      
    #download data
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    #the path for the idx directory
    save_path = io_3d.latest_filename_data(save_path) + "/"
    #make a new data idx directory
    os.makedirs(save_path)
    #download only if the download flag is set
    if args.download:
      model.io_3d.download_zip(config_3d.URL_DATA, config_3d.DATA_PATH)
    dtype = get_data_idx(args.type)
    tr, val = get_ds(load_path, config_3d.TRAIN_SIZE, config_3d.VAL_SIZE, config_3d.TRAIN_SIZE+config_3d.VAL_SIZE, config_3d.PT_SAMPLE)
    
    #save datasets to the same directory as the model
    model.io_3d.save_ds(tr, 'tr', save_path)
    model.io_3d.save_ds(val, 'val', save_path)
    
  else:
    #get the path to the target model
    model_path = config_3d.MODEL_PATH + args.name + '/'
    model_obj_path = model_path + args.name + '.pt'
    #update the data idx
    data_idx = 0 if args.data is None else args.data
  #if new model, download and create dataset, make new model
  if args.command == 'new':
    #first check if the dataset exists
    if not os.path.exists(data_path):
      print(f"Dataset not found in path {data_path}: create dataset using [data] argument")
      exit()
    #check if the dataset exists
    if os.path.exists(model_path):
      print(f"Model {args.name} already exists in {model_path}: use [train] argument to do additional training or create a model with different name")
      exit()
    save_path = config_3d.DATA_PATH + args.name
    #make new  model
    model_idx = 1 if not args.model else args.model
    init_grid = ops_3d.init_grid_3d(config_3d.GRID_SIZE).view(-1).to(config_3d.DEVICE)
    model = network_3d.ALIGNet_3d(args.name, config_3d.GRID_SIZE, config_3d.VOX_SIZE, init_grid, model_idx).to(config_3d.DEVICE)
    
    #make a directory to save model and dataset
    if not os.path.exists(config_3d.MODEL_PATH + args.name):
        os.makedirs(config_3d.MODEL_PATH + args.name)
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs')
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs/loss_graphs')
        os.makedirs(config_3d.MODEL_PATH + args.name + '/outputs/images')
    tr, val = load_ds(data_idx)
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    train_model(args, tr, val, model_path)
    
  #common operation for train and valid 
  elif args.command == 'train' or args.command == 'valid':
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
    model = torch.load(model_obj_path, map_location=torch.device('cpu')).to(config_3d.DEVICE)
    model = model.to(config_3d.DEVICE)
    tr, val = load_ds(data_idx)
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    train_model(args, tr, val, model_path)
    
  if args.command == 'valid':
    #load the previous model
    model = torch.load(model_obj_path, map_location=torch.device('cpu')).to(config_3d.DEVICE)
    #path to save the visualized image 
    image_path = model_path + 'outputs/images/'
    image_path = io_3d.latest_filename_png(image_path)
    #load the neccessary datasets
    tr, val = load_ds(data_idx)
    tr.to(config_3d.DEVICE)
    val.to(config_3d.DEVICE)
    #get the target and val sets
    #visualize if specified
    #pointcloud
    #DEPRECATED
    if args.visualize == 'pointcloud':
      print("This function is deprecated and is no longer supported => aborting..")
      exit()
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      result_checker.update()
      result_checker.get_pointcloud_from_mesh(config_3d.PT_SAMPLE)
      result_checker.visualize(datatype=2, sample=config_3d.NUM_SAMPLE, save_path=image_path)
      
    #visualize mesh
    #DEPRECATED
    elif args.visualize == 'mesh':
      print("This function is deprecated and is no longer supported => aborting..")
      exit()
      #for mesh visualization, we need to set get_mesh=True
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, get_src_mesh=True, get_tar_pt=False)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      #we will retrieve the original src mesh representation and apply deformation directly
      result_checker.update(get_mesh=True)
      result_checker.warp_mesh()
      result_checker.get_mesh_from_vox()
      result_checker.visualize(datatype=1, sample=config_3d.NUM_SAMPLE, save_path=image_path)
      
    #a custom visualization routine designed for research purposes 
    elif args.visualize == 'custom':
      #for mesh visualization, we need to set get_mesh=True, get_for_grid=True
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL, get_src_mesh=True, get_tar_pt=True)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      #we will retrieve the original src mesh representation and apply deformation directly
      result_checker.update(get_src_mesh=True, get_tar_pt = True, get_for_grid=True)
      result_checker.warp_mesh()
      result_checker.visualize(datatype=1, sample=config_3d.NUM_SAMPLE, save_path=image_path)
    #default: voxel visualization
    else:
      val_tar_dl, val_src_dl = datasets.get_val_dl_3d(
          val, config_3d.TARGET_PROPORTION_VAL, config_3d.BATCH_SIZE, config_3d.VOX_SIZE, config_3d.AUGMENT_TIMES_VAL)
      result_checker = validate_3d.result_checker_3d(model, val_tar_dl, val_src_dl)
      result_checker.update()
      result_checker.visualize(datatype=0, sample=config_3d.NUM_SAMPLE, save_path=image_path)
 
    
      
      
      
    
    
