#This file defines the visualization routines
#pointcloud and voxels visualization routines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random

#Some other 3d visualization routines using open 3d
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader

#visualize pointclouds
#pointcloud: shape(minibatch, 3)
def visualize_pointclouds(pointcloud):
  fig = go.Figure(data=[go.Scatter3d(
      x=pointcloud[0][:,0],
      y=pointcloud[0][:,1],
      z=pointcloud[0][:,2],
      mode='markers',
      marker=dict(
          size=1,
          colorscale='Viridis',   # choose a colorscale
          opacity=0.8
      )
  )])
  fig.show()


#visualize voxel (single voxel)
#voxel: shape(x, y, z)
def visualize_voxels(voxel):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.voxels(voxel, edgecolor="k")
  plt.show()

def visualize_results_3d_vox(src_batch, tar_batch, tar_est, index):
  for i in index:
    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')
    ax1.set_title('Source Voxel')
    fig.tight_layout() 
    ax1.voxels(src_batch[i], edgecolor='k')
    ax2 = fig.add_subplot(222,projection='3d')
    ax2.set_title('Target Voxel')
    fig.tight_layout() 
    ax2.voxels(tar_batch[i], edgecolor='k')
    ax3 = fig.add_subplot(223,projection='3d')
    ax3.set_title('Estimated Voxel')
    fig.tight_layout() 
    ax3.voxels(tar_est[i],  edgecolor='k')

#BUG: Not working (though mesh data are sound)
#+random sampling from mesh dataset not working
def visualize_results_3d_mesh(src_batch, tar_batch, tar_est, index):
  print(f"visualize_results_3d_mesh: shape of src_batch-{len(src_batch)}")
  print(f"visualize_results_3d_mesh: index-{index}")
  print(f"visualize_results_3d_mesh: src_batch[0]: {src_batch[0]}")
  print(f"visualize_results_3d_mesh: src_batch type: {type(src_batch)}")
  for i in range(len(src_batch)-1):
    src = src_batch[0]
    tar = tar_batch[0]
    est = tar_est[0]
    print(f"visualize_results_3d_mesh: {src.verts_list()}")
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}],
           [{'type': 'mesh3d'}, {'type': 'mesh3d'}]])
    fig.add_trace(
      go.Mesh3d(
        x=torch.stack(src.verts_list(), dim=0)[:,0], y=torch.stack(src.verts_list(), dim=0)[:,1], z=torch.stack(src.verts_list(), dim=0)[:,2],
        i=torch.stack(src.faces_list(), dim=0)[:,0], j=torch.stack(src.faces_list(), dim=0)[:,1], k=torch.stack(src.faces_list(), dim=0)[:,2],
      ),
      row=1, col=1
    )
    fig.add_trace(
      go.Mesh3d(
        x=torch.stack(tar.verts_list(), dim=0)[:,0], y=torch.stack(tar.verts_list(), dim=0)[:,1], z=torch.stack(tar.verts_list(), dim=0)[:,2],
        i=torch.stack(tar.faces_list(), dim=0)[:,0], j=torch.stack(tar.faces_list(), dim=0)[:,1], k=torch.stack(tar.faces_list(), dim=0)[:,2],
      ),
      row=2, col=1
    )
    fig.add_trace(
      go.Mesh3d(
        x=torch.stack(est.verts_list(), dim=0)[:,0], y=torch.stack(est.verts_list(), dim=0)[:,1], z=torch.stack(est.verts_list(), dim=0)[:,2],
        i=torch.stack(est.faces_list(), dim=0)[:,0], j=torch.stack(est.faces_list(), dim=0)[:,1], k=torch.stack(est.faces_list(), dim=0)[:,2],
      ),
      row=1, col=2
    )
    #fig.update_layout(height=600, width=800, title_text="source, target, and target_estimate meshes")
    fig.show()
    #break

def visualize_results_3d_pt(src_batch, tar_batch, tar_est, index):
  for i in index:
    src = src_batch[i]
    tar = tar_batch[i]
    est = tar_est[i]
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
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
      row=1, col=1
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
      row=2, col=1
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
      row=1, col=2
    )
    fig.update_layout(height=600, width=800, title_text="source, target, and target_estimate pointclouds")
    fig.show()
  
  
#visualize the results given source, target, and target estimate images
#save_img is the path to save the img
#datatype: 0-voxel, 1-mesh, 2-pointcloud
def visualize_results_3d(src_batch,  tar_batch, tar_est, datatype=0, batch_index=0, sample=None, save_path=None):
  src_batch = src_batch[batch_index].to(CPU)
  tar_batch = tar_batch[batch_index].to(CPU)
  tar_est = tar_est[batch_index].to(CPU)
  batch = len(src_batch)
  index = np.arange(batch-1)
  #sample specific number of elements from the batch if specified
  if sample:
    index = np.random.choice(index, sample)
  index = np.sort(index)
  print(f"visualize_results_3d: index-{index}")
  #the case for when datatype is voxel
  if datatype==0:
    visualize_results_3d_vox(src_batch, tar_batch, tar_est, index)
  elif datatype==1:
    visualize_results_3d_mesh(src_batch, tar_batch, tar_est, index)
  elif datatype==2:
    visualize_results_3d_pt(src_batch, tar_batch, tar_est, index)

  if save_path:
    plt.savefig(save_path, format='png')
  return



#given any type of iterables -- list, arrays, tensors, datasets, etc-- sample data from the set and visualize
#iter_return denotes how many output the iterable returns for each iteration
def visualize_sample(iterable, sample_size, iter_return = 1):
  iter_length = len(iterable)
  sample_index = []
  i = 0
  #append n sample indexes to the list sample_index
  while(not i == len(sample_size)-1):
    index = np.random.randint(mask_size_low, mask_size_high)
    if index not in sample_index:
      sample_index.append[index]
  #create a list to store the sample data
  data_list = []
  for i in range(sample_size):
    index = sample_index[i]
    if iter_return == 1:
      data = iterable[index]
      data_list.append(data)
    else: 
      data_src, data_tar = iterable[index]
      data_list.append([data_src, data_tar])
  o3d.visualization.draw_geometries(data_list)
