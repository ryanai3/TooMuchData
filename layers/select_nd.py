import torch
from torch import autograd
from torch.autograd import Variable

def normalize_nd(x, idxs, c):
  if x.dim() == 3:
    b, c, v = x.size()
    x = x.view(b * c, v)
  if idxs.dim() == 2:
    idxs = idxs[:, 0] * c + idxs[:, 1]
  return x, idxs

def select_nd(x, idxs, c=None):
  norm_x, norm_idxs = normalize_nd(x, idxs, c)  
  res = norm_x[norm_idxs]
  return norm_x[norm_idxs]

def assign_nd(x, idxs, value, c=None):
  norm_x, norm_idxs = normalize_nd(x, idxs, c)
  norm_x[norm_idxs] = value
      
class SelectND(autograd.Function):

  def __init__(self, idxs, debug=False):
    self.idxs = idxs

        
  def forward(self, x):
    self.save_for_backward(x)
    selected = select_nd(x, self.idxs)
    return selected

  def backward(self, grad_output):
    x = self.saved_tensors[0]
    gx = torch.zeros(x.size()).cuda()
    assign_nd(gx, self.idxs, grad_output)
    return gx
   
class SelectiveUpdate(autograd.Function):

  def __init__(self, idxs):
    self.idxs = idxs

  def forward(self, x, update):
    new_x = x.clone()
    assign_nd(new_x, self.idxs, update)
    return new_x
        
  def backward(self, grad_output):
    gx = grad_output.clone()
    gupdate = select_nd(gx, self.idxs)
    assign_nd(gx, self.idxs, 0)
    return gx, gupdate
