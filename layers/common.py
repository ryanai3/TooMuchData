import torch
from torch import autograd

class InPlaceFunction(autograd.Function):

  def forward(self, x):
    self.save_for_backward(x)
    return x

  def backward(self, gx):
    return gx
