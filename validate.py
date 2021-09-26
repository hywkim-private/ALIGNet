#This cell contains properties necessary for validation 
import torch 
import torchvision
import numpy as np
import augment
import load_data
import matplotlib
import matplotlib.pyplot as plt

#visualize the results given source, target, and target estimate images
#save_img is the path to save the img
def visualize_results(source_image,  target_image, target_estimate, save_image=None):
  batch, _, _ = source_image.shape
  fig, ax = plt.subplots(batch, 3, figsize=(20,20))
  for i in range(batch):
    #ax[i,0].set_title('source_image')
    ax[i,0].imshow(source_image[i], cmap='gray')
    #ax[i,1].set_title('target_image')
    ax[i,1].imshow(target_image[i], cmap='gray')
    #ax[i,2].set_title('target_estimate')
    ax[i,2].imshow(target_estimate[i], cmap='gray')
  if not save_image == None:
    plt.savefig(save_image)
  return

#given source and target validation  dataloaders, perform all validation operations
#save_img is the path to save the img
@torch.no_grad()
def validate(model, source_dl, target_dl, grid_size, visualize = False, get_loss = False, save_image=None, checker_board=False):
  #we will run everything, save the results, and visualize the images all at once
  source = []
  target = []
  target_est = []
  loss_list = []
  target_iter = iter(target_dl)
  source_iter = iter(source_dl)
  for i in range(len(target_dl)):
    target_image = next(target_iter)
    source_image = next(source_iter)
    target_image = torch.FloatTensor(target_image).squeeze(dim=1).to(DEVICE)
    source_image = torch.FloatTensor(source_image).squeeze(dim=1).to(DEVICE)
    input_image = torch.stack([source_image, target_image])
    input_image  = input_image.permute([1,0,2,3])
    if checker_board:
      target_image = apply_checkerboard(target_image, 128)
      source_image = apply_checkerboard(source_image, 128)
    tar_est, diff_grid = model.forward(input_image, source_image)
    tar_est = tar_est.squeeze(dim=1)
    if get_loss: 
      L2_Loss_ = L2_Loss(target_image, tar_est)
      L_TV_Loss_ = L_TV_Loss(diff_grid, 8, 1)
      loss = L_TV_Loss_ + L2_Loss_
      loss_list.append(loss)
    source_image = source_image.to(torch.device('cpu')).detach().numpy()
    target_image = target_image.to(torch.device('cpu')).detach().numpy()
    tar_est = tar_est.squeeze().to(torch.device('cpu')).detach().numpy()
    if visualize:
      source.append(source_image)
      target.append(target_image)
      target_est.append(tar_est)
  if visualize:
    for i in range(len(source)):
      source_image = source[i]
      target_image = target[i]
      target_estimate = target_est[i]
      visualize_results(source_image, target_image, target_estimate, save_image)
  if get_loss:
    avg_loss = sum(loss_list)/len(loss_list)
    print(f"Average Loss: {avg_loss}")
    return avg_loss


#this class stores necessary in order to check and prevent overfitting of the model
class overfit_checker():
  def __init__(self, model, valid_tar, valid_src):
    self.model = model
    self.valid_tar = valid_tar
    self.valid_src = valid_src
    self.train_loss = []
    self.val_loss = []
    #the difference bw train and valid set
    self.tr_val_gap = []
    #the differential list of tr_val_gap
    self.tr_val_diff = []
    self.epoch = 0
    #if this flag is set to True, we will halt the operation
    self.halt = False
    self.graph = None
  
  #run the validate function and update the tr_val_gap parameter
  def update(self,train_loss):
    val_loss = validate(model, self.valid_src, self.valid_tar, GRID_SIZE, get_loss = True)
    self.val_loss.append(val_loss)
    self.train_loss.append(train_loss)
    self.tr_val_gap.append(val_loss - train_loss)
    tr_val_diff = 0
    if not self.epoch == 0:
      tr_val_diff = self.tr_val_gap[len(self.val_loss)-1] - self.tr_val_gap[len(self.val_loss)-2]
      self.tr_val_diff.append(tr_val_diff)
    self.epoch += 1
    if tr_val_diff > 0:
      print('Halt flag has been raised')
      self.halt = True 

  def print_graph(self):
    plt.plot(self.train_loss, 'bo', label='Training Loss')
    plt.plot(self.val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()