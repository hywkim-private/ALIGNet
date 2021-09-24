
#function source: https://teamdable.github.io/techblog/PyTorch-Autograd
def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  #info.append(f'tensor({str(tensor)})')
  return ' '.join(info)