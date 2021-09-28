import config
import torch 
from torch.autograd import Variable
import torch.optim as optim
import model
import validate
import grid_helper
from loss_functions import L2_Loss, L_TV_Loss

  
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
      aug_batch = torch.tensor(aug_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config.DEVICE)
      tar_batch = torch.tensor(tar_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config.DEVICE)
      src_batch = torch.tensor(src_batch, dtype=torch.float32, requires_grad=True).squeeze(dim=1).to(config.DEVICE)
      input_image = torch.stack([src_batch, aug_batch])
      input_image  = input_image.permute([1,0,2,3])

      #run the network
      tar_est, diff_grid = model.forward(input_image, src_batch)
      tar_est = tar_est.squeeze(dim=1)
      L2_Loss_ = L2_Loss(tar_batch, tar_est)

      L_TV_Loss_ = L_TV_Loss(diff_grid, 8, 1e-3)
  
      loss = L_TV_Loss_ + L2_Loss_
      loss.backward()
      #L_TV_Loss_.backward(retain_graph=True)
      optimizer.step()
      optimizer.zero_grad()
      loss_list.append(loss)
  return loss_list

def run_model(model,source_loader, target_loader, grid_size, result_checker=None, graph_loss=False):
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  epoch_loss = []
  for i in range(config.EPOCHS):
    loss_list = run_epoch(model, optimizer, source_loader, target_loader, grid_size)
    avg_epoch_loss = sum(loss_list) / len(loss_list)
    avg_epoch_loss = avg_epoch_loss.item()
    print(f'Loss in Epoch {i}: {avg_epoch_loss}')
    if not result_checker == None:
      result_checker.update(avg_epoch_loss)
  if (not result_checker == None) and graph_loss:
    print("printing graph..")
    result_checker.print_graph(save_path = './' + model.name + '/outputs/loss_graphs/')


  