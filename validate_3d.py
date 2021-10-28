#This cell contains properties necessary for validation 
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt
from pytorch3d import ops
from preprocess import convert_type_3d as conv
import config_3d
from model import loss_3d
from utils.vis_3d import visualize_results_3d

#TODO: THE MESH REPRESENTATION OF THE INPUT DATA IS NOT STORED IN DATALOADERS IN THE FIRST PLACE
#MUST MODIFY THE DATALOADERS IN ORDER TO RETURN A PACKED (MESH, VOXEL) TUPLE IF SPECIFIED BY AN ARGUMENT (DONE)


#TODO: FOR NOW, THE FUNCTION AUG_DATASETS_3D GETS SRC DATALOADER THAT RETURNS BOTH VOXEL AND MESH IF 
#THE DATASET IS KNOWN TO BE A VALID DATASET
#HOWEVER, FOR THE SAKE OF MEMORY, MUST ALSO CONSIDER GETTING SUCH DATALOADER ONLY IF SPECIFIED BY THE VISUALIZE PARAMETER
#THAT IS, WE WILL ONLY GET MESH REPRESENTATIONS (AND THUS WARP) ONLY WHEN WE ARE GOINGI TO VISUALIZE THE VALIDATION RESULTS 


#this class stores necessary in order to check and validate results of the model
class result_checker_3d():
  def __init__(self, model, valid_tar, valid_src):
    self.model = model
    self.valid_tar = valid_tar
    self.valid_src = valid_src
    self.val_loss = []
    self.train_loss = []
    self.epoch = 0
    #if this flag is set to True, we will halt the operation
    self.graph = None
    self.avg_loss = 0
    self.updated = False
    self.mesh = False

  #run the validate function and update the tr_val_gap parameter
  #include a train loss if youre validating for a training loop
  def update(self,train_loss=None):
    tar_list, src_list, est_list, diff_grid_list, def_grid_list, avg_loss = validate_dl_3d(self.model, self.valid_src, self.valid_tar, config_3d.GRID_SIZE)
    self.avg_loss = avg_loss
    self.tar_list = tar_list
    self.src_list = src_list
    self.est_list = est_list
    #TODO: estimation in mesh form -- apply deformation directly to the mesh
    self.est_mesh_list = None 
    self.diff_grid_list = diff_grid_list
    self.def_grid_list = def_grid_list
    self.val_loss.append(avg_loss)
    self.train_loss.append(train_loss)
    self.epoch += 1
    self.updated = True
    
  #apply the deformation directly to mesh, and save it in est_mesh_list
  def warp_mesh(self):
    
    
  #after updating the voxelized results, get their mesh representations 
  #save lists of voxelized tar, src, est data
  def get_mesh(self):
    if self.updated == False:
      print("result_checker_3d.get_mesh: Error-call update() before get_mesh()")
      return
    self.tar_mesh = []
    self.src_mesh = []
    self.est_mesh = []
    for i in range(len(self.tar_list)):
      self.tar_mesh.append(ops.cubify(self.tar_list[i], 1))
      self.src_mesh.append(ops.cubify(self.src_list[i], 1))
      self.est_mesh.append(ops.cubify(self.est_list[i], 1))
    self.mesh=True

  def get_pointcloud(self, num_samples):
    if self.mesh == False:
      #call get_mesh if it hasn't been called
      self.get_mesh()
    self.tar_pt =[]
    self.src_pt =[]
    self.est_pt =[]
    for i in range(len(self.tar_list)):
      self.tar_pt.append(conv.PointCloud(self.tar_mesh[i], num_samples=num_samples))
      self.src_pt.append(conv.PointCloud(self.src_mesh[i], num_samples=num_samples))
      self.est_pt.append(conv.PointCloud(self.est_mesh[i], num_samples=num_samples))

  #visualize the results in images 
  #batch_index specifies which batch to visualize
  #datatype : 0-voxel, 1-mesh, 2-pointcloud
  def visualize(self, datatype=0, batch_index=0, sample=None, save_path=None):
    if datatype == 0:
      visualize_results_3d(self.src_list, self.tar_list, self.est_list, datatype, batch_index, sample, save_path)
    elif datatype == 1:
      visualize_results_3d(self.src_mesh, self.tar_mesh, self.est_mesh, datatype, batch_index, sample, save_path)
    elif datatype == 2:
      visualize_results_3d(self.src_pt, self.tar_pt, self.est_pt, datatype, batch_index, sample, save_path)

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
    

#given source and target validation  dataloaders, perform all validation operations
#save_img is the path to save the img
@torch.no_grad()
def validate_3d(model, source_image, target_image, grid_size):
  input_image = torch.stack([source_image, target_image])
  input_image  = input_image.permute([1,0,2,3,4])
  diff_grid = model.forward(input_image)
  #we will use def grid to apply warp directly to mesh
  def_grid = model.cumsum(diff_grid)
  tar_est = model.warp(def_grid, source_image)
  tar_est = tar_est.squeeze(dim=1)
  #calculate the loss for the image
  loss = loss_3d.get_loss_3d(target_image, tar_est, diff_grid, config_3d.GRID_SIZE, config_3d.VOX_SIZE)
  return tar_est, diff_grid, def_grid, loss

#run loop for the dataloaders to run the validate() operation
@torch.no_grad()
def validate_dl_3d(model, source_dl, target_dl, grid_size):
  target_iter = iter(target_dl)
  source_iter = iter(source_dl)
  #we will also create lists for tar and src since dataloaders may shuffle them in random orders
  target_list = []
  source_list = []
  est_list = []
  diff_grid_list = []
  def_grid_list = []
  loss_list = []
  for i in range(len(target_dl)):
    target_image = next(target_iter)
    source_image = next(source_iter)
    target_image = torch.FloatTensor(target_image).squeeze(dim=1).to(config_3d.DEVICE)
    source_image = torch.FloatTensor(source_image).squeeze(dim=1).to(config_3d.DEVICE)
    input_image = torch.stack([source_image, target_image])
    input_image  = input_image.permute([1,0,2,3,4])
    tar_est, diff_grid, def_grid, loss = validate_3d(model, source_image, target_image, grid_size)
    target_list.append(target_image)
    source_list.append(source_image)
    est_list.append(tar_est)
    diff_grid_list.append(diff_grid)
    def_grid_list.append(def_grid)
    loss_list.append(loss)
  avg_loss_ = loss_3d.avg_loss_3d(loss_list)
  return target_list, source_list, est_list, diff_grid_list, def_grid_list, avg_loss_
  

