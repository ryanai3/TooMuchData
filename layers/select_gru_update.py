from torch import autograd
from torch.autograd import Variable

class SelectND(autograd.Function):

  def __init(self, idxs):
    if idxs.dim() == 2:
      ...
  
  def forward(self, x):
    if x.dim() == 3:
      self.orig_size = x.size()
      b, c, v = self.orig_size
      x = x.view(b * c, v)
    self.save_for_backward(x)
    selected = x[self.idxs]
    return selected

  def backward(self, grad_output):
    x = self.saved_tensors[0]
    gx = torch.zeros(x.size()).cuda()
    gx[self.idxs] = grad_output
    return gx.view(self.orig_size)
   
class SelectiveUpdate(autograd.Function):

  def __init__(self, idxs, ):
    self.idxs = idxs
    update.dim() == 2:

  def forward(self, x, update):
    self.save_for_backward(x, update)
    if x.dim() == 3:
      b, c, v = x.size()
      x = x.view(b * c, v)
      
    vx = x.view(
    new_x = torch.zeros(x.size()).cuda()
    new_x = x.data
    new_x[self.idxs] = update
    return new_x
        
  def backward(self, grad_output):
    gx = grad_output.clone()
    gupdate = torch.zeros(grad_output.size()).cuda()
    gx[self.idxs] = 0
    gupdate[self.idxs] = grad_output
    return gx, gupdate
