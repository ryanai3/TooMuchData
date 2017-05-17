import torch
from torch import autograd
from layers.common import InPlaceFunction

class InPlaceDetach(InPlaceFunction):

  def __init__(self, selector):
    self.selector = selector

  def backward(self, grad_output):
    grad_output[self.selector] = 0
    return grad_output

class InPlaceZero(InPlaceFunction):

  def __init__(self, selector):
    self.selector = selector

  def forward(self, x):
    self.save_for_backward(x)
    x[self.selector] = 0
    return x

class InPlaceZeroDetach(InPlaceDetach, InPlaceZero):
  
  def __init__(self, selector):
    self.selector = selector
