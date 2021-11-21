#This cell contains properties necessary for validation 
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt
from pytorch3d import ops
from pytorch3d.structures import Meshes, Pointclouds
from preprocess import convert_type_3d as conv
import config_3d
from model import loss_3d, ops_3d
from utils.vis_3d import visualize_results_3d, visualize_pointclouds, visualize_mesh



#TODO: THE MESH REPRESENTATION OF THE INPUT DATA IS NOT STORED IN DATALOADERS IN THE FIRST PLACE
#MUST MODIFY THE DATALOADERS IN ORDER TO RETURN A PACKED (MESH, VOXEL) TUPLE IF SPECIFIED BY AN ARGUMENT (DONE)


#TODO: FOR NOW, THE FUNCTION AUG_DATASETS_3D GETS SRC DATALOADER THAT RETURNS BOTH VOXEL AND MESH IF 
#THE DATASET IS KNOWN TO BE A VALID DATASET
#HOWEVER, FOR THE SAKE OF MEMORY, MUST ALSO CONSIDER GETTING SUCH DATALOADER ONLY IF SPECIFIED BY THE VISUALIZE PARAMETER
#THAT IS, WE WILL ONLY GET MESH REPRESENTATIONS (AND THUS WARP) ONLY WHEN WE ARE GOINGI TO VISUALIZE THE VALIDATION RESULTS 

#TODO: ****MAKE THE VISUALIZE MESH FUNCTIONALITY WORK******
#THE PROBLEM FOR VISUALIZATION MIGHT BE SOLVED BY USING THE PADDED LIST BUT NOT SURE
#FIRST, FIX THE UNIT TEST CODE SNIPPET IN GOOGLE COLAB

#TODO: MAKE SURE THE VOXEL VISUALIZATION/PT VIS WORKS
#THE INTERPOLATION IS GIVING FUCKING WEIRD RESULTS=>CHECK THE INTERPOLATION FUNCTION (ACTUALLY THIS MIGHT BE FROM LACK OF TRAINING-->SOME LOOKS GOOD)
#TAR_EST IN MESH VISUALIZATION, SAMPLED FROM PT, LOOKS  WEIRD. => MAYBE ALSO JUST GET DIRECTLY FROM THE DATALOADER
#UNIFY THE FUNCTIONALITY FOR VISUALIZING MESH => DON'T GIVE TOO MANY OPTIONS, LIKE VISUALIZING FROM FULLY SAMPLED-POINTS MESHS=>WE DON'T NEED THEM HONESTLY


#MAKE SURE ALL IMAGES ARE ALIGNED PROPERLY=>CHECK AXIS
#FIX THE PT MASK FUNCTION IN GOOGLE COLAB
#SEE POINTCLOUD VISUALIZATION=>SOMETHING FUCKING WEIRD GOING ON
#FIND THE APPROPRIATE "CONVERSION RANGE" FROM (0,32) TO (-1,1) AND FIX THE MASK, INTERPOLATION FUNCTION
#CHECK THE VOXELIZATION SCHEME => TURN OFF NORMALIZATON
  
  
#TODO: UNIFY THE METHOD OF DATA CONCATENATION IN CONVERT_PY , DATASETS_PY  


#TODO:IMPLEMENT THE "FLIP" FUNCTIONALITY FOR VISUALIZATION
#DIVERSIFY THE MODEL


#TODO: THE DATA VISUALIZATION YIELDS OVERLAPPING IMAGES THOUGH IT SHOULDN'T => FIX 


#https://stackoverflow.com/questions/59327021/how-to-plot-a-2d-structured-mesh-in-matplotlib



#TWO POSSIBLE POINTS OF ERROR: 1. INTEPOLATE_3D 2. CONVERTING VOXEL => MESH DEF_GRID


#ONLY THING LEFT TO CHECK: IF THE TWO DEF_GRIDS ARE EQUAL (RIGHT AFTER RETURNS AND BEFORE INTERPOLATING)
#IF NOT, PRBLM WITH THE INTERPOLATE

#__init__ input
#model: ALIGNet_3d model
#valid_tar: target dataset
#valid-src: source dataset

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
  #inputs
  #self:self
  #train_loss: whether or not to get the train set loss
  #get_src_mesh: get the mesh representation for src dataset
  #get_tar_pt: get the  pointcloud representation for the target dataset
  def update(self, train_loss=None, get_src_mesh=False, get_tar_pt=False, get_for_grid=False):
    #run the validation
    self.validate_dl_3d(self.model, self.valid_src, self.valid_tar, config_3d.GRID_SIZE, get_src_mesh, get_tar_pt, get_for_grid=get_for_grid)
    #set the get_mesh pararmeter to True in order to use the warp_mesh functionality 
    #the visualize() function will still work without update(get_mesh=True). But will only get the mesh/pointcloud data as sampled from their voxel counterparts
    self.src_mesh_ori = None
    if get_src_mesh: 
      src_mesh_ori_list = []
      #TODO:UPGRADE THIS FUNCTIONALITY USING THE MESH.CONCAT FUNCTION(?)
      #we need "list of verts and faces" to create Mesh object that stores all the meshes we have
      for mesh in self.src_mesh_list:
        #make src_mesh_list to a Mesh datatype
        vert_list = []
        face_list = []
        for m in mesh:
          vert = torch.stack(m.verts_list(), dim=0)[0]
          face = torch.stack(m.faces_list(), dim=0)[0]
          vert_list.append(vert)
          face_list.append(face)
        src_mesh_ori = Meshes(vert_list, face_list)
        src_mesh_ori_list.append(src_mesh_ori)
      self.src_mesh_ori = src_mesh_ori_list
    
    if get_tar_pt:
      tar_pt_ori_list = []
      #we need "list of verts and faces" to create Mesh object that stores all the meshes we have
      for point in self.tar_pt_list:
        #make src_mesh_list to a Mesh datatype
        point_list = []
        for p in point:
          point = torch.stack(p.points_list(), dim=0)[0]
          point_list.append(point)
        tar_pt_ori = Pointclouds(point_list)
        tar_pt_ori_list.append(tar_pt_ori)
      self.tar_pt_ori = tar_pt_ori_list
    self.epoch += 1
    self.updated = True
    
  #apply the deformation directly to mesh, and save it in est_mesh_list
  def warp_mesh(self):
    #if the src_mesh list doesn't exist, raise error
    if self.src_mesh_ori == None:
      print("result_checker_3d.warp_mesh: ERROR-src_mesh_list is not initialized."
      "Make sure to call update with get_src_mesh=True in order to retrieve the original mesh representation of source dataset.")
      return
    #in order for warp_mesh to work in sync with visualization, we first need to interpolate meshes individually,
    #then append them altogether in the Mesh datatype
    else: 
      def_mesh_list = []
      for mesh, for_grid in zip(self.src_mesh_ori, self.for_grid_list):
        #def grid of shape (N,C,D,H,W)
        vertice_list = []
        face_list = []
        #for each batch of inputs
        for i in range(len(mesh)):
          #get verts and faces for a single mesh
          m = mesh[i]
          m_verts = m.verts_list()[0]
          m_faces = m.faces_list()[0]
          m_faces = np.stack(m_faces, axis=0)
          m_verts = np.stack(m_verts, axis=0)
          #m_verts = np.flip(m_verts, 1)
          g = for_grid[i]
          #interpolate the single mesh with the forward warp
          deformed_verts = ops_3d.interpolate_3d_mesh(m_verts, g, config_3d.VOX_SIZE)
          
          #make faces and verts into Tensors so it can make up for the Mesh datatype
          m_faces = torch.Tensor(m_faces)
          deformed_verts = torch.Tensor(deformed_verts)
          #get the "list of tensors" for verts and faces
          vertice_list.append(deformed_verts)
          face_list.append(m_faces)
        #turn list of faces and verts tensors into Mesh datatype 
        deformed_mesh = Meshes(vertice_list, face_list)
        def_mesh_list.append(deformed_mesh)
    #deformed_mesh is a list of meshes, grouped in batches
    self.deformed_mesh = def_mesh_list
    
  #after updating the voxelized results, get their mesh representations
  #save lists of voxelized tar, src, est data
  def get_mesh_from_vox(self):
    if self.updated == False:
      print("result_checker_3d.get_mesh: Error-call update() before get_mesh()")
      return
    self.tar_mesh = []
    self.src_mesh = []
    self.est_mesh = []
    for i in range(len(self.tar_list)):
      #tar_mesh, src_mesh, est_mesh consists of lists of batches
      self.tar_mesh.append(ops.cubify(self.tar_list[i], 1))
      self.src_mesh.append(ops.cubify(self.src_list[i], 1))
      self.est_mesh.append(ops.cubify(self.est_list[i], 1))
    self.mesh=True


  def get_pointcloud_from_mesh(self, num_samples):
    if self.mesh == False:
      #call get_mesh if it hasn't been called
      self.get_mesh_from_vox()
    self.tar_pt =[]
    self.src_pt =[]
    self.est_pt =[]
    for i in range(len(self.tar_list)):
      self.tar_pt.append(conv.PointCloud(self.tar_mesh[i], num_samples, config_3d.DEVICE))
      self.src_pt.append(conv.PointCloud(self.src_mesh[i], num_samples, config_3d.DEVICE))
      self.est_pt.append(conv.PointCloud(self.est_mesh[i], num_samples, config_3d.DEVICE))


  #visualize the results in images 
  #batch_index specifies which batch to visualize
  #datatype : 0-voxel, 1-src(mesh)-tar(pointclud)-tar_est(mesh)
  def visualize(self, datatype=0, batch_index=0, sample=None, save_path=None):
    if datatype == 0:
      visualize_results_3d(self.src_list, self.tar_list, self.est_list, self.def_grid_list, datatype, batch_index, sample, save_path)
    elif datatype == 1: 
      #FOR NOW WE WILL MANUALLY SET DATATYPE=> TODO:SYNCHRONIZE THE NAMING OF DATATYPES
      visualize_results_3d(self.src_mesh_ori, self.tar_pt_ori, self.deformed_mesh, self.for_grid_list, 3, batch_index, sample, save_path)


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
  def validate_3d(self, model, source_image, target_image, grid_size, get_for_grid=False):
    input_image = torch.stack([source_image, target_image])
    input_image  = input_image.permute([1,0,2,3,4])
    diff_grid = model.forward(input_image)
    #we will use def grid to apply warp directly to mesh
    def_grid = model.cumsum(diff_grid)
    tar_est = model.warp(def_grid, source_image)
    tar_est = tar_est.squeeze(dim=1)
    #convert the backward ward into forward ward
    g = def_grid.clone() 
    g = g.to(config_3d.CPU)
    g = g.detach().numpy()
    
    #calculate the loss for the image
    init_grid = ops_3d.init_grid_3d(grid_size).to(config_3d.DEVICE)
    loss = loss_3d.get_loss_3d(target_image, tar_est, diff_grid, init_grid, config_3d.GRID_SIZE, config_3d.VOX_SIZE)
    if get_for_grid:
      i=0
      for g_ in g:
        x_fr, y_fr, z_fr = ops_3d.convert_to_forward_warp(g_[0],g_[1],g_[2])
        for_grid = np.stack([x_fr, y_fr, z_fr])
        g[i] = for_grid
        i+=1
      for_grid = g
      return tar_est, diff_grid, def_grid, for_grid, loss
    return tar_est, diff_grid, def_grid, loss
  #run loop for the dataloaders to run the validate() operation
  #if get_mesh param is set to true, also return the mesh representation of src dataset (returns 7 results instead of 6 )
  @torch.no_grad()
  def validate_dl_3d(self, model, source_dl, target_dl, grid_size, get_src_mesh=False, get_tar_pt=False, get_for_grid=False):
    target_iter = iter(target_dl)
    source_iter = iter(source_dl)
    #we will also create lists for tar and src since dataloaders may shuffle them in random orders
    target_list = []
    source_list = []
    est_list = []
    diff_grid_list = []
    def_grid_list = []
    for_grid_list = []
    loss_list = []
    #if visualize is True, initialize the list for src mesh
    if get_src_mesh:
      source_mesh_list = []
    if get_tar_pt:
      target_pt_list = []
    for i in range(len(target_dl)):
      #if get_src_mesh is true get the mesh representation of the source dataset
      #parameters that check if src and tar datasets are retrieved
      src_init = False
      tar_init = False
      if get_src_mesh:
        #if get_src_mesh, the dataloader will return both src_vox and src_mesh
        source, source_mesh = next(source_iter)
        source_mesh_list.append(source_mesh)
        src_init = True
    
      #if get_tar_pt is true, get the pointclouds of the tar dataset
      if get_tar_pt:
        #if get_tar_pt is True the dataloader will return both vox and pt
        target, target_pt = next(target_iter)
        target_pt_list.append(target_pt)
        tar_init = True 
      if src_init == False:
        source = next(source_iter)
      if tar_init == False:
        target = next(target_iter)
        
      target = torch.FloatTensor(target).squeeze(dim=1).to(config_3d.DEVICE)
      source = torch.FloatTensor(source).squeeze(dim=1).to(config_3d.DEVICE)
      if get_for_grid:
        tar_est, diff_grid, def_grid, for_grid, loss = self.validate_3d(model, source, target, grid_size,get_for_grid=get_for_grid)
        for_grid_list.append(for_grid)

      else: 
        tar_est, diff_grid, def_grid, loss = self.validate_3d(model, source, target, grid_size)
      target_list.append(target)
      source_list.append(source)
      est_list.append(tar_est)
      diff_grid_list.append(diff_grid)
      def_grid_list.append(def_grid)
      if get_for_grid:
        for_grid_list.append(for_grid)
      loss_list.append(loss)
    avg_loss_ = loss_3d.avg_loss_3d(loss_list)
    #update th class parameterss
    self.avg_loss = avg_loss_
    self.tar_list = target_list
    self.src_list = source_list
    self.est_list = est_list
    self.diff_grid_list = diff_grid_list
    self.def_grid_list = def_grid_list
    self.for_grid_list = for_grid_list
    self.val_loss.append(avg_loss_)
    #also update src_mesh and tar_pt if specified
    if get_src_mesh:
      self.src_mesh_list = source_mesh_list
    if get_tar_pt:
      self.tar_pt_list = target_pt_list
    if get_for_grid: 
      self.for_grid_list = for_grid_list


  

