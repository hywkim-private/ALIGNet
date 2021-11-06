import glob
#function source: https://teamdable.github.io/techblog/PyTorch-Autograd
def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  #info.append(f'tensor({str(tensor)})')
  return ' '.join(info)
  

#a function that parses through the file and returns the filename of the highest index number
def latest_filename(file_path):
  file_list = []
  for file in glob.glob(file_path + '/*.png', recursive=True):
    str_list = str(file).split('/')
    file = str_list[len(str_list)-1]
    file = file[0]
    file = int(file)
    file_list.append(file)
  if not len(file_list) == 0:
    max_num = max(file_list) + 1
  else:
    max_num = 0
  filename = file_path  + str(max_num) + '.png'
  return filename
    
