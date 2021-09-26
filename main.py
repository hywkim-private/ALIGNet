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
  print(torch.__version__)
  print(torch.version.cuda)
  config.initialize_config()
  #argument parser
  ap = argparse.ArgumentParser()
  subap = ap.add_subparsers(dest='command')
  new = subap.add_parser('new')
  val = subap.add_parser('val')
  data = subap.add_parser('data')

  #if -n argument is given, create a new model
  new.add_argument("-f", "--filepath", type=str,  help="Specify the filepath to create newmodel")
  new.add_argument("-n", "--name", required=True, type=str,  help="Specify the name of the new model")
  new.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  new.add_argument("-i", "--iter", type=str, help="number of times to run the training loop, default: 10") 
  #download and save new dataset
  data.add_argument("-ty", "--type", required=True, type=str,  help="Specify the dataset to use for the new model")
  


  args = ap.parse_args()
  #if new data, download data, create, and save
  if args.command == 'data': 
    #first, check for existing directory and if not, create a new directory to store data 
    if args.type == 'vase':
      if not os.path.exists(config.FILE_PATH_VASE):
        os.makedirs(config.FILE_PATH_VASE)
      load_data.download_data(config.URL_VASE, config.VASE)
          
    if args.type == 'plane':
      if not os.path.exists(config.FILE_PATH_PLANE):
        os.makedirs(config.FILE_PATH_PLANE)    
      load_data.download_data(config.URL_PLANE,config.PLANE)
      
  #if new model, download and create dataset, make new model
  if args.command == 'new':
    save_path = config.FILE_PATH + args.name
    #make new  model
    model = model.ALIGNet(config.GRID_SIZE)
    #make a directory to save model and dataset
    if not os.path.exists(config.MODEL_PATH + args.name):
        os.makedirs(config.MODEL_PATH + args.name)
        os.makedirs(config.MODEL_PATH + args.name+'/datasets')
    #decide and load which type of data to use
    if args.type == 'vase':
      #make datasets 
      tr, val, test = load_data.get_datasets(config.VASE)
    elif args.type == 'plane ':  
      #make datasets
      tr, val, test = load_data.get_datasets(config.PLANE)
    #save datasets to the same directory as the model
    load_save.save_ds(tr, 'tr', config.MODEL_PATH + args.name +'/datasets/')
    load_save.save_ds(val, 'val', config.MODEL_PATH + args.name +'/datasets/')
    load_save.save_ds(test, 'test', config.MODEL_PATH + args.name +'/datasets/')
    
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
    
    
    
  