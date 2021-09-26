import config
import torch 
from torch.autograd import Variable
import torch.optim as optim
import model
import validate
   
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
  
#run the model for a single epoch
#data_loader can either be a single data_loader or a iterable list of data loaders (if so, source loader should also be a list of data loaders)
def run_epoch(model, optimizer,  source_loader, data_loader, grid_size):
  #raise error if input types dont match
  if not type(data_loader) == type(source_loader):
    print('ERROR: data_laoder and source_loader must be of the same dtype')
    return None
  #if single value is given, convert data and source loader to list
  elif not type(data_loader) is list:
    data_loader = [data_loader]
    source_loader = [source_loader]
  loss_list = []
  #iterate over the list
  for j in range(len(data_loader)):
    tar_loader = data_loader[j]
    src_loader = source_loader[j]
    #raise error if the length of tar and src loader dont match
    if not len(tar_loader) == len(src_loader):
      print('ERROR: The length of tar_loader and src_loader does not match')
      return
    #define the iterables  
    tar_iter = iter(tar_loader)
    src_iter = iter(src_loader)
    for k in range(len(tar_loader)-1):
      aug_batch, tar_batch = next(tar_iter)
      src_batch = next(src_iter)
      aug_batch = torch.tensor(aug_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(DEVICE)
      tar_batch = torch.tensor(tar_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(DEVICE)
      src_batch = torch.tensor(src_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(DEVICE)
      input_image = torch.stack([src_batch, aug_batch])
      input_image  = input_image.permute([1,0,2,3])

      #run the network
      tar_est, diff_grid = model.forward(input_image, src_batch)
      tar_est = tar_est.squeeze(dim=1)
      L2_Loss_ = L2_Loss(tar_batch, tar_est)

      #L2_Loss_.retain_grad()
      L_TV_Loss_ = L_TV_Loss(diff_grid, 8, 1e-3)
      #print(L2_Loss_)
      #print(diff_grid)
      loss = L_TV_Loss_ + L2_Loss_
      loss.backward()
      #L_TV_Loss_.backward(retain_graph=True)
      optimizer.step()
      optimizer.zero_grad()
      loss_list.append(loss)
  return loss_list

def run_model(model,source_loader, target_loader, grid_size, overfit_checker=None, graph_loss=False):
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  epoch_loss = []
  for i in range(EPOCHS):
    loss_list = run_epoch(model, optimizer, source_loader, target_loader, grid_size)
    avg_epoch_loss = sum(loss_list) / len(loss_list)
    print(f'Loss in Epoch {i}: {avg_epoch_loss}')
    if not overfit_checker == None:
      overfit_checker.update(avg_epoch_loss)
  if (not overfit_checker == None) and graph_loss:
    overfit_checker.print_graph()


  