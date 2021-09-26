import config
import model
import validate
import train
import load_data
import load_save
import argparse
import os

if __name__ == '__main__':
  config.initialize_config()
  #argument parser
  ap = argparse.ArgumentParser()
  subap = ap.add_subparsers(dest='command')
  new = subap.add_parser('new')
  validate = subap.add_parser('validate')
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
      if not os.path.exists(config.FILE_PATH_VASE)
          os.makedirs(config.FILE_PATH_VASE
    if args.type == 'plane':
      if not os.path.exists(config.FILE_PATH_PLANE)
          os.makedirs(config.FILE_PATH_PLANE)    
  #if new model, download and create dataset, make new model
  if args.command == 'new':
    save_path = config.FILE_PATH + args.name
    #make new  model
    model = model.ALIGNet(config.GRID_SIZE)
    #decide and load which type of data to use
    if args.type == 'vase':
      tr, val, test = load_data.load_ds(config.FILE_PATH_VASE)
    elif args.type == 'plane ':  
      tr, val, test = load_data.load_ds(config.FILE_PATH_PLANE, ds_index=1)
    #save the iter variable as default or specified by the user
    iter = 10
    if args.iter: 
      iter = args.iter
    #make an overfit check if specified by config 
    overfit_check = None 
    if config.OVERFIT_CHECK:
       #make the valid dataset
      val_tar, val_src, val_tar_aug, val_src_aug = load_data.get_val_dl(tr, val, test)
      overfit_check = validate.overfit_checker(model,val_tar_vase, val_src_vase)
    train(model, iter, tr, val, test, overfit_check, args.name, train_mode=config.TRAIN_MODE, graph_loss=config.GRAPH_LOSS)
  