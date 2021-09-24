def L2_Loss(target_image, warped_image):
  sum_check = torch.norm(target_image-warped_image, p=2)/BATCH_SIZE
  sum_check = sum_check/(128*128)
  L2_Loss = sum_check
  return L2_Loss

def L_TV_Loss(diff_grid, grid_size, lambda_):
  #create the identity differential grid  
  batch, _,w,h = diff_grid.shape
  diff_i_grid = init_grid(grid_size)
  diff_i_grid = diff_i_grid.view(2,grid_size,grid_size)
  diff_i_grid_x = diff_i_grid[0]
  diff_i_grid_y = diff_i_grid[1]
  L_TV_Loss = torch.norm(diff_grid[:,0] - diff_i_grid_x, p=1)/batch + torch.norm(diff_grid[:,1] - diff_i_grid_y, p=1)/batch
  L_TV_Loss = L_TV_Loss / (w*h) 
  L_TV_Loss = L_TV_Loss * lambda_
  return L_TV_Loss

#define the alignnet model
def get_conv(grid_size):
  model = nn.Sequential (
      nn.MaxPool2d (2),
      nn.Conv2d (2, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 2),
      nn.ReLU(),
      nn.MaxPool2d (2),
      nn.Conv2d (20, 20, 5),
      nn.ReLU(),
   
  )
  return model

"""#forward hook for the alignet model
#primarily used to calculate the l_tv regularization term
class ALIGNet_Hook():
  def __init__(self, model):
    self.hook = model.register_forward_hook(hook_fn)
  def hook_fn(model, input, output):
    """
class warp_layer(nn.Module):
  def __init__(self, grid_size, checker_board = False):
    super().__init__()
    self.upsampler = nn.Upsample(size = [IMAGE_SIZE, IMAGE_SIZE], mode = 'bilinear')
    self.grid_offset_x = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True) 
    self.grid_offset_y = torch.tensor(float(-1-2/(grid_size-1)), requires_grad=True)
    self.grid_offset_x = nn.Parameter(self.grid_offset_x)
    self.grid_offset_y = nn.Parameter(self.grid_offset_y)
    self.grid_size = grid_size

  def forward(self, x, src_batch, checker_board=False):
    #perform the cumsum operation to restore the original grid from the differential grid
    x = cumsum_2d(x, self.grid_offset_x, self.grid_offset_y)
    #Upsample the grid_size x grid_size warp field to image_size x image_size warp field
    x = self.upsampler(x)
    x = x.permute(0,2,3,1)
    if checker_board:
      source_image = apply_checkerboard(src_batch, IMAGE_SIZE)
    #calculate target estimation
    x = nn.functional.grid_sample(src_batch.unsqueeze(0).permute([1,0,2,3]), x, mode='bilinear')
    return x

#a layer that ensure axial monotinicity
class axial_layer(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.grid_size = grid_size
  def forward(self, x):
    #enforce axial monotinicity using the abs operation
    x = torch.abs(x)
    batch, grid = x.shape
    x = x.view(batch, 2,self.grid_size,self.grid_size)
    return x

#define the convolutional + linear layers
class conv_layer(nn.Module):
  def __init__(self, grid_size):
    super().__init__()
    self.conv = get_conv(grid_size)
    self.flatten = nn.Flatten()
    self.linear1 = nn.Sequential(nn.Linear(80,20),nn.ReLU(),)
    self.linear2 = nn.Linear(20, 2*grid_size*grid_size)
    self.linear2.bias = nn.Parameter(init_grid(grid_size).view(-1))
    self.linear2.weight.data.fill_(float(0))
  def forward(self, x):
    x = self.conv(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x

#define the model class
class ALIGNet(nn.Module):
  def __init__(self, grid_size, checker_board=False):
    super().__init__()
    self.conv_layer = conv_layer(grid_size)
    self.warp_layer = warp_layer(grid_size)
    self.axial_layer = axial_layer(grid_size)
    self.checker_board = checker_board
  def forward(self, x, src_batch=None, warp=True):
    x = self.conv_layer(x)
    x = self.axial_layer(x)
    diff_grid = x
    if warp:
      x = self.warp_layer(x, src_batch, self.checker_board)
    return x, diff_grid

