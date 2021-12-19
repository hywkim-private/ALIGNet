#This cell contains properties necessary for validation 
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import config
import skimage
import load_data
from utils import etc
from utils.loss_functions import L2_Loss, L_TV_Loss, get_loss, avg_loss
import matplotlib
import matplotlib.pyplot as plt

#functionality that creates a checkerboard mask the size of grid
def get_checkerboard(image_size):
  cb = skimage.data.checkerboard()
  cb = torch.FloatTensor(cb)
  cb = cb.unsqueeze(0).unsqueeze(0)
  #Upsample the grid_size x grid_size warp field to image_size x image_size warp field
  #We will use bilinear upsampling
  upsampler = nn.Upsample(size = [image_size, image_size], mode = 'bilinear')
  upsampled_cb = upsampler(cb)
  upsampled_cb = upsampled_cb.squeeze().to(config.DEVICE)
  return upsampled_cb

def apply_checkerboard(batch, imsize):
  cb = get_checkerboard(imsize)
  checker_batch = batch[:] * cb
  return checker_batch


#given source and target validation  dataloaders, perform all validation operations
#save_img is the path to save the img
@torch.no_grad()
def validate(model, source_image, target_image, grid_size, image_size, checker_board=False):
  input_image = torch.stack([source_image, target_image])
  input_image  = input_image.permute([1,0,2,3])
  tar_est, diff_grid = model.forward(input_image, source_image)
  tar_est = tar_est.squeeze(dim=1)
  #calculate the loss for the image
  loss = get_loss(target_image, tar_est, grid_size,  diff_grid, image_size, config.DEVICE)
  #apply checkerboard and run warp again if specified by checker_board
  if checker_board:
    source_ck = apply_checkerboard(source_image, 128)
    tar_est_ck = model.warp(diff_grid, source_ck)
    return tar_est_ck, diff_grid, loss
  return tar_est, diff_grid, loss

#run loop for the dataloaders to run the validate() operation
@torch.no_grad()
def validate_dl(model, source_dl, target_dl, grid_size, image_size, checker_board=False):
  target_iter = iter(target_dl)
  source_iter = iter(source_dl)
  #we will also create lists for tar and src since dataloaders may shuffle them in random orders
  target_list = []
  source_list = []
  est_list = []
  grid_list = []
  loss_list = []
  for i in range(len(target_dl)):
    target_image = next(target_iter)
    source_image = next(source_iter)
    target_image = torch.FloatTensor(target_image).squeeze(dim=1).to(config.DEVICE)
    source_image = torch.FloatTensor(source_image).squeeze(dim=1).to(config.DEVICE)
    input_image = torch.stack([source_image, target_image])
    input_image  = input_image.permute([1,0,2,3])
    tar_est, diff_grid, loss = validate(model, source_image, target_image, grid_size, image_size, checker_board)
    if checker_board:
      target_image = apply_checkerboard(target_image, 128)
      source_image = apply_checkerboard(source_image, 128)
    target_list.append(target_image)
    source_list.append(source_image)
    est_list.append(tar_est)
    grid_list.append(diff_grid)
    loss_list.append(loss)
  avg_loss_ = avg_loss(loss_list)
  return target_list, source_list, est_list, grid_list, avg_loss_
  


#print the visual results of the validation run
def print_image(tar_list, src_list, est_list, save_path=None):
  for i in range(len(tar_list)):
    target_image = tar_list[i]
    source_image = src_list[i]
    tar_est = est_list[i]
    #send images to cpu and detatch in order to convert them into numpy objects
    source_image = source_image.to(torch.device('cpu')).detach().numpy()
    target_image = target_image.to(torch.device('cpu')).detach().numpy()
    target_estimate = tar_est.squeeze().to(torch.device('cpu')).detach().numpy()
    visualize_results(source_image, target_image, target_estimate, save_path)
    return 
 
 
#the parallel multiprocessor version of the function visualize results
#visualize the results given source, target, and target estimate images
#save_img is the path to save the img
def visualize_results(source_image,  target_image, target_estimate, save_path=None):
  batch, _, _ = source_image.shape
  fig, ax = plt.subplots(batch, 3, figsize=(20,20))
  for i in range(batch):
    #ax[i,0].set_title('source_image')
    ax[i,0].imshow(source_image[i], cmap='gray')
    #ax[i,1].set_title('target_image')
    ax[i,1].imshow(target_image[i], cmap='gray')
    #ax[i,2].set_title('target_estimate')
    ax[i,2].imshow(target_estimate[i], cmap='gray')
  if save_path:
    plt.savefig(save_path, format='png')
  return



#this class stores necessary in order to check and validate results of the model
class result_checker():
  def __init__(self, model, valid_tar, valid_src, checker_board=False):
    self.model = model
    self.valid_tar = valid_tar
    self.valid_src = valid_src
    self.val_loss = []
    self.train_loss = []
    self.epoch = 0
    #if this flag is set to True, we will halt the operation
    self.halt = False
    self.graph = None
    self.checker_board = checker_board 
    self.avg_loss = 0
  
  #run the validate function and update the tr_val_gap parameter
  #include a train loss if youre validating for a training loop
  def update(self, train_loss=None):
    tar_list, src_list, est_list, grid_list, avg_loss = validate_dl(self.model, self.valid_src, self.valid_tar, config.GRID_SIZE, config.IMAGE_SIZE,checker_board=self.checker_board)
    self.avg_loss = avg_loss
    self.tar_list = tar_list
    self.src_list = src_list
    self.est_list = est_list
    self.grid_list = grid_list
    self.val_loss.append(avg_loss)
    self.train_loss.append(train_loss)
    self.epoch += 1 
  
  #visualize the results in images 
  def visualize(self, save_path):
    print_image(self.tar_list, self.src_list, self.est_list, save_path=save_path)
  
 #update the loss value after running the model
  #print the graph of the current loss results->save to path if specified
  def print_graph(self, save_path=None):
    plt.plot(self.train_loss, 'bo', label='Training Loss')
    plt.plot(self.val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()
    if save_path:
      plt.savefig(etc.latest_filename(save_path),format='png')
    plt.close()
    

    
#a function that creates result checker and runs validation 
def run_validation(model, source_loader, target_loader, grid_size, save_path, get_loss=True, visualize=True, checker_board=True):
  #create a result checker object
  result = result_checker(model, target_loader, source_loader, checker_board)
  #run and update the parameters in the result checker object
  result.update()
  if visualize:
    print(f"printing image results to {save_path}")
    result.visualize(save_path)
  if get_loss:
    print(f"The average loss of validation set is: {result.avg_loss}")
    
    
    