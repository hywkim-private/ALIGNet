#CORE HELPER FUNCTIONS

#initialize the differential grid
#the parameter learn offset will define whether or not to learn the offset values during training
def init_grid(grid_size):
  #spacing of the grid
  #-1 is because we have a1 = -1 (and thus there are grid_size - 1 "spacing" grids)
  delta = 2/(grid_size-1)
  np_grid = np.arange(grid_size, dtype=float)
  np_grid = np.full_like(np_grid,float(delta))
  ts_grid_x = torch.FloatTensor(np_grid).to(DEVICE)
  ts_grid_y = torch.FloatTensor(np_grid).to(DEVICE)
  diff_i_grid_y, diff_i_grid_x = torch.meshgrid(ts_grid_x,ts_grid_y)
  diff_grid = torch.stack([diff_i_grid_x, diff_i_grid_y])
  diff_grid = diff_grid.view(2*grid_size*grid_size)
  return diff_grid

#perform cumsum operation on a 2d batch of inputs
#takes in grid tensors of shape batch x 2 x grid x grid 
#return grid tensors of shape batch x 2 x grid x grid 
def cumsum_2d(grid, grid_offset_x, grid_offset_y):
  batch_size, dim, grid_1, grid_2 = grid.shape

 
  Integrated_grid_x = torch.cumsum(grid[:,0], dim = 2) + grid_offset_x
  Integrated_grid_y = torch.cumsum(grid[:,1], dim = 1) + grid_offset_y
  Integrated_grid = torch.stack([Integrated_grid_x, Integrated_grid_y])
  Integrated_grid = Integrated_grid.permute([1,0,2,3])

  return Integrated_grid

#visualize image given a data loader
def visualize_image(data_loader, plot_size):
  fig, ax = plt.subplots(plot_size, plot_size, figsize=(20,20))
  x = 0
  y = 0
  for i,(aug_batch,tar_batch) in enumerate(data_loader):
    for k,image in enumerate(aug_batch):
      image = image.squeeze()
      if x >= plot_size:
        x = 0
        y += 1
      if y >= plot_size:
        return
      ax[x,y].imshow(image, cmap='gray')
      x += 1
  