#This file defines the visualization routines
#pointcloud and voxels visualization routines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random

#Some other 3d visualization routines using open 3d
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

def visualize_results_3d_vox(src_batch, tar_batch, tar_est, batch_index, index):
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

#visualize sets of src, tar, and est meshes according to the sampled index
#input
#src_batch, tar_batch, tar_est: pytorch3d mesh datatype
#index: list of sample indexes
def visualize_results_3d_mesh(src_batch, tar_batch, tar_est, batch_index, index):
  print(f"length of the tar_batch: {len(src_batch)}")
  print(f"length of the tar_batch: {len(tar_est)}")
  #access the specific batch to visualize using the batch_index
  src_verts = src_batch[batch_index].verts_list()
  print(f"length of the src_verts: {len(src_verts)}")
  src_faces = src_batch[batch_index].faces_list()
  tar_verts = tar_batch[batch_index].verts_list()
  tar_faces = tar_batch[batch_index].faces_list()
  est_verts = tar_est[batch_index].verts_list()
  est_faces = tar_est[batch_index].faces_list()
  for i in index:
    src_vert = src_verts[i]
    src_face = src_faces[i]
    tar_vert = tar_verts[i]
    tar_face = tar_faces[i]
    est_vert = est_verts[i]
    est_face = est_faces[i]
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}],
           [{'type': 'mesh3d'}, {'type': 'mesh3d'}]])
    fig.add_trace(
      go.Mesh3d(
        x=src_vert[:,0], y=src_vert[:,1], z=src_vert[:,2],
        i=src_face[:,0], j=src_face[:,1], k=src_face[:,2],
      ),
      row=1, col=1
    )
    fig.add_trace(
      go.Mesh3d(
        x=tar_vert[:,0], y=tar_vert[:,1], z=tar_vert[:,2],
        i=tar_face[:,0], j=tar_face[:,1], k=tar_face[:,2],
      ),
      row=2, col=1
    )
    fig.add_trace(
      go.Mesh3d(
        x=est_vert[:,0], y=est_vert[:,1], z=est_vert[:,2],
        i=est_face[:,0], j=est_face[:,1], k=est_face[:,2],
      ),
      row=1, col=2
    )
  return fig 

def visualize_results_3d_pt(src_batch, tar_batch, tar_est, batch_index, index):
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
  
  return fig 
  
  
#visualize the results given source, target, and target estimate images
#save_img is the path to save the img
#datatype: 0-voxel, 1-mesh, 2-pointcloud
def visualize_results_3d(src_batch,  tar_batch, tar_est, datatype=0, batch_index=0, sample=None, save_path=None):
  
  batch_len = len(tar_batch[0])
  index = np.arange(batch_len)
  #sample specific number of elements from the batch if specified
  if sample:
    index = np.random.choice(index, sample, replace=False)
  index = np.sort(index)
  #the case for when datatype is voxel
  if datatype==0:
    visualize_results_3d_vox(src_batch, tar_batch, tar_est, batch_index, index)
  elif datatype==1:
    fig = visualize_results_3d_mesh(src_batch, tar_batch, tar_est, batch_index,  index)
  elif datatype==2:
    visualize_results_3d_pt(src_batch, tar_batch, tar_est, batch_index, index)
  #datatype 0  uses the matplotlib library
  if save_path and datatype == 0:
    print(f"Visualize_results_3d: Image saved to path {save_path}")
    plt.savefig(save_path, format='png')
  #datatype 1 and 2 uses a plotly dependency
  elif save_path:
    print(f"Visualize_results_3d: Image saved to path {save_path}")
    fig.write_image(save_path + '.png')
  return



