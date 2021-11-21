#This file defines the visualization routines
#pointcloud and voxels visualization routines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random

#Some other 3d visualization routines using open 3d
import torch
from torch.utils.data import Dataset, DataLoader

#given a 3d array of shape (x,y,z) and a sample step,
#sample elements indiced by the step
def sample_step(ts, step):
  new_len = int(len(ts)/step)
  index = np.linspace(0, len(ts)-1, new_len)
  index = np.rint(index)
  index = index.astype(np.int)
  new_ts = ts[index]
  new_ts = new_ts[:, index]
  new_ts = new_ts[:, :, index]
  return new_ts

#given meshgrids along x,y,z axis all of shape (x,y,z)
#return thea list of plotly figures that depicts the grid
def get_meshgrid_fig(X,Y,Z):
  lines = []
  grid_range = len(X)
  for idx in range(grid_range):
    for i in range(grid_range):
      x=X[:,idx,i]
      y=Y[:,idx,i]
      z=Z[:,idx,i]
      lines.append(go.Scatter3d(
      x=x,
      y=y,
      z=z,
      mode='lines',
      line = dict(
        color = "#bcbd22",
        width = 1.5
        ),
      connectgaps=True))
    for i in range(grid_range):
      x=X[idx,:,i]
      y=Y[idx,:,i]
      z=Z[idx,:,i]
      lines.append(go.Scatter3d(
      x=x,
      y=y,
      z=z,
      mode='lines',
      line = dict(
        color = "#bcbd22",
        width = 1.5
        ),
      connectgaps=True))
    for i in range(grid_range):
      x=X[idx,i,:]
      y=Y[idx,i,:]
      z=Z[idx,i,:]
      lines.append(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        line = dict(
        color = "#bcbd22",
        width = 1.5
        ),
        connectgaps=True))
  return lines
    

#make matplotlib LineCollection object for visualizing voxels
#input: def_grid of shape (x,y,z)=> X,Y, Z
#return: np array of shape (x*y*z,3)
def get_linecol(X,Y,Z):
  lines = []
  grid_range = len(X)
  for idx in range(grid_range):
    for i in range(grid_range):
      x=X[:,idx,i]
      y=Y[:,idx,i]
      z=Z[:,idx,i]
      lines.append(np.stack((x,y,z), axis=1))
    for i in range(grid_range):
      x=X[idx,:,i]
      y=Y[idx,:,i]
      z=Z[idx,:,i]
      lines.append(np.stack((x,y,z), axis=1))
    l3=[]
    for i in range(grid_range):
      x=X[idx,i,:]
      y=Y[idx,i,:]
      z=Z[idx,i,:]
      lines.append(np.stack((x,y,z), axis=1))
  lines = np.stack(lines)

  return lines
      
#given a (x,y,z) grid in rang (-1,1)
#remap the coordinates to shape (0, max_coord)
def remap_local(grid, max_coord):
  num_el = len(grid)
  grid = np.add(grid,np.array([1]))
  grid = np.multiply(grid, np.array([max_coord/2]))
  return grid
  
#visualize pointclouds
#pointcloud: shape(minibatch, 3)
def visualize_pointclouds(pointcloud):
  fig = go.Figure(data=[go.Scatter3d(
      x=pointcloud[0][:,0],
      y=pointcloud[0][:,1],
      z=pointcloud[0][:,2],
  )])
  return fig
#visualize voxel (single voxel)
#voxel: shape(x, y, z)
def visualize_voxels(voxel):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.voxels(voxel, edgecolor="k")
  plt.show()


def visualize_mesh(vert, face):
  #access the specific batch to visualize using the batch_index
  fig = go.Figure(data=[go.Mesh3d(
      x=vert[:,0], y=vert[:,1], z=vert[:,2],
      i=face[:,0], j=face[:,1], k=face[:,2],
    )])
  return fig 
  
  
def visualize_results_3d_vox(src_batch, tar_batch, tar_est, def_grids, batch_index, index):
  src_batch = src_batch[batch_index]
  tar_batch = tar_batch[batch_index]
  tar_est = tar_est[batch_index]
  def_grids = def_grids[batch_index]
  fig = plt.figure(figsize=(15*len(index), 60))
  for idx,i in enumerate(index):
    #make the deformation grid lines 
    def_grid = def_grids[idx]
    #downsample the deformation grids
    def_x = sample_step(def_grid[0],8).cpu().numpy()
    def_y = sample_step(def_grid[1],8).cpu().numpy()
    def_z = sample_step(def_grid[2],8).cpu().numpy()
    #TODO: CHANGE THIS TO PROPER INTERPOLATION (converting coordinates)
    def_x = remap_local(def_x, 32)
    def_y = remap_local(def_y, 32)
    def_z = remap_local(def_z, 32)

    def_line = get_linecol(def_x, def_y, def_z)
    
    #define the regular (undeformed grid)
    ls = np.linspace(0,32,4)
    #make the regular grid lines
    reg_x, reg_y, reg_z = np.meshgrid(ls,ls,ls)
    reg_line = get_linecol(reg_x, reg_y, reg_z)
   
    ax1 = fig.add_subplot(len(index),3,idx*3+1,projection='3d', computed_zorder=True)
    ax1.set_title('Source Voxel')
    ax1.set_xlabel("x")
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.tight_layout() 
    ax1.voxels(src_batch[i], edgecolor='k', zorder=0)
    for k in range(len(reg_line)):
      ax1.plot(reg_line[k,:,0],reg_line[k,:,1],reg_line[k,:,2],color="red", zorder=100+k)

    ax2 = fig.add_subplot(len(index), 3, idx*3+2,projection='3d', computed_zorder=True)
    ax2.set_title('Target Voxel')
    ax2.set_xlabel("x")
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    fig.tight_layout() 
    ax2.voxels(tar_batch[i], edgecolor='k', zorder=0)
    for k in range(len(reg_line)):
      ax2.plot(reg_line[k,:,0],reg_line[k,:,1],reg_line[k,:,2],color="red", zorder=100+k)

    ax3 = fig.add_subplot(len(index), 3, idx*3+3,projection='3d', computed_zorder=True)
    ax3.set_title('Estimated Voxel')
    ax3.set_xlabel("x")
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    fig.tight_layout()
    ax3.voxels(tar_est[i],  edgecolor='k', zorder=0)
    for k in range(len(def_line)):
      ax3.plot(def_line[k,:,0],def_line[k,:,1],def_line[k,:,2],color="red", zorder=100+k)
  return fig



#visualize sets of src, tar, and est meshes according to the sampled index
#input
#src_batch, tar_batch, tar_est: pytorch3d mesh datatype
#index: list of sample indexes
def visualize_results_3d_mesh(src_batch, tar_batch, tar_est, batch_index, index):
  #access the specific batch to visualize using the batch_index
  src_verts = src_batch[batch_index].verts_list()
  src_faces = src_batch[batch_index].faces_list()
  tar_verts = tar_batch[batch_index].verts_list()
  tar_faces = tar_batch[batch_index].faces_list()
  est_verts = tar_est[batch_index].verts_list()
  est_faces = tar_est[batch_index].faces_list()
  spec_rows = [{'type':'mesh3d'}] * 3
  spec = [spec_rows] * len(index)
  fig = make_subplots(rows=len(index), cols=3, specs=spec)
  for idx,i in enumerate(index):
    src_vert = src_verts[i]
    src_face = src_faces[i]
    tar_vert = tar_verts[i]
    tar_face = tar_faces[i]
    est_vert = est_verts[i]
    est_face = est_faces[i]
    fig.add_trace(
      go.Mesh3d(
        x=src_vert[:,0], y=src_vert[:,1], z=src_vert[:,2],
        i=src_face[:,0], j=src_face[:,1], k=src_face[:,2],
      ),
      row=idx+1, col=1
    )
    fig.add_trace(
      go.Mesh3d(
        x=tar_vert[:,0], y=tar_vert[:,1], z=tar_vert[:,2],
        i=tar_face[:,0], j=tar_face[:,1], k=tar_face[:,2],
      ),
      row=idx+1, col=2
    )
    fig.add_trace(
      go.Mesh3d(
        x=est_vert[:,0], y=est_vert[:,1], z=est_vert[:,2],
        i=est_face[:,0], j=est_face[:,1], k=est_face[:,2],
      ),
      row=idx+1, col=3
    )
  fig.update_layout(height=300*len(index), width=900, title_text="Source, Target, and Target Estimate Mesh")
  return fig 

def visualize_results_3d_pt(src_batch, tar_batch, tar_est, batch_index, index):
  src_batch = src_batch[batch_index].points_list()
  tar_batch = tar_batch[batch_index].points_list()
  tar_est = tar_est[batch_index].points_list()
  spec_rows = [{'type':'scatter3d'}] * 3
  spec = [spec_rows] * len(index)
  fig = make_subplots(rows=len(index), cols=3, specs=spec)
  for idx, i in enumerate(index):
    src = src_batch[i]
    tar = tar_batch[i]
    est = tar_est[i]
    fig.add_trace(
      go.Scatter3d(
        name='Source',
        x=src[:,0],
        y=src[:,1],
        z=src[:,2],
        mode='markers',
        marker=dict(
            size=1,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
      ),
      row=idx+1, col=1
    )
    fig.add_trace(
      go.Scatter3d(
        name='Target',
        x=tar[:,0],
        y=tar[:,1],
        z=tar[:,2],
        mode='markers',
        marker=dict(
            size=1,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
      ),
      row=idx+1, col=2
    )
    fig.add_trace(
      go.Scatter3d(
        name='Target Estimate',
        x=est[:,0],
        y=est[:,1],
        z=est[:,2],
        mode='markers',
        marker=dict(
            size=1,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
      ),
      row=idx+1, col=3
    )
  fig.update_layout(height=600*len(index), width=1200, title_text="Source, Target, and Target_Estimate Pointclouds")
  return fig 
  
  
#customized visualization-visualize in the form of src(Mesh), tar(PointClouds), tar-est(Mesh)
#input
#src_mesh, tar_pt, tar_est_mesh
#batch_index: the index of batch from which to sample data
#index: list of sample indexes
def visualize_results_3d_custom(src_mesh, tar_pt, tar_est_mesh, def_grids, batch_index, index):
  
  src_mesh = src_mesh[batch_index]
  tar_pt = tar_pt[batch_index]
  tar_est_mesh = tar_est_mesh[batch_index]
  def_grids = def_grids[batch_index]
  src_mesh_verts = src_mesh.verts_list()
  src_mesh_faces = src_mesh.faces_list()
  tar_pt_points = tar_pt.points_list()
  tar_est_mesh_verts = tar_est_mesh.verts_list()
  tar_est_mesh_faces = tar_est_mesh.faces_list()
  spec_rows = [{'type':'mesh3d'}, {'type':'scatter3d'}, {'type':'mesh3d'}] 
  spec = [spec_rows] * len(index)
  fig = make_subplots(rows=len(index), cols=3, specs=spec)
  for idx, i in enumerate(index):
    src_vert = src_mesh_verts[i]
    src_face = src_mesh_faces[i]
    tar_pt_point = tar_pt_points[i]
    tar_est_mesh_vert = tar_est_mesh_verts[i]
    tar_est_mesh_face = tar_est_mesh_faces[i]
    def_grid = def_grids[i]

    #define the regular (undeformed grid)
    #downsample the deformation grids
    def_i = sample_step(def_grid[0],8)
    def_j = sample_step(def_grid[1],8)
    def_k = sample_step(def_grid[2],8)
    def_lines = get_meshgrid_fig(def_k, def_j, def_i)
    #define the deformation grid
    ls = np.linspace(-1,1,4)
    ls = torch.Tensor(ls).cpu()
    reg_i, reg_j, reg_k = torch.meshgrid(ls,ls,ls)
    reg_lines = get_meshgrid_fig(reg_i, reg_j, reg_k)
    src_trace = go.Mesh3d(
          x=src_vert[:,0],
          y=src_vert[:,1],
          z=src_vert[:,2],
          i=src_face[:,0],
          j=src_face[:,1],
          k=src_face[:,2],
    )
    tar_trace = go.Scatter3d(
        x=tar_pt_point[:,0],
        y=tar_pt_point[:,1],
        z=tar_pt_point[:,2],
        mode='markers',
        marker=dict(
            size=1,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
    tar_est_trace = go.Mesh3d(
      x=tar_est_mesh_vert[:,0],
      y=tar_est_mesh_vert[:,1],
      z=tar_est_mesh_vert[:,2],
      i=tar_est_mesh_face[:,0],
      j=tar_est_mesh_face[:,1],
      k=tar_est_mesh_face[:,2],
    )
    fig.add_trace(
      src_trace,
      row=idx+1, col=1
    )
    fig.add_traces(
      reg_lines,
      rows=idx+1, cols=1
    )
    fig.add_trace(
      tar_trace,
      row=idx+1, col=2
    )
    fig.add_traces(
      reg_lines,
      rows=idx+1, cols=2
    )
    fig.add_trace(
      tar_est_trace,
      row=idx+1, col=3
    )
    fig.add_traces(
      def_lines,
      rows=idx+1, cols=3
    )
  fig.update_layout(height=400*len(index), width=1800, title_text="Source Meshes, Target PointClouds, and Target_Estimate Meshes")
  return fig 
  
  

#visualize the results given source, target, and target estimate images
#save_img is the path to save the img
#datatype: 0-voxel, 1-mesh, 2-pointcloud
def visualize_results_3d(src_batch,  tar_batch, tar_est, def_grid, datatype=0, batch_index=0, sample=None, save_path=None):
  batch_len = len(tar_batch[0])
  index = np.arange(batch_len)
  #sample specific number of elements from the batch if specified
  if sample:
    index = np.random.choice(index, sample, replace=False)
  index = np.sort(index)
  #the case for when datatype is voxel
  if datatype==0:
    visualize_results_3d_vox(src_batch, tar_batch, tar_est, def_grid, batch_index, index)
  elif datatype==1:
    fig = visualize_results_3d_mesh(src_batch, tar_batch, tar_est, batch_index,  index)
  elif datatype==2:
    fig = visualize_results_3d_pt(src_batch, tar_batch, tar_est, batch_index, index)
  elif datatype==3:
    fig = visualize_results_3d_custom(src_batch, tar_batch, tar_est, def_grid, batch_index, index)
    
  #datatype 0  uses the matplotlib library
  if save_path and datatype == 0:
    print(f"Visualize_results_3d: Image saved to path {save_path}")
    plt.savefig(save_path, format='png')
  #datatype 1 and 2 uses a plotly dependency
  elif save_path:
    print(f"Visualize_results_3d: Image saved to path {save_path}")
    fig.write_image(save_path)

  return



