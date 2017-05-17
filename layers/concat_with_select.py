import torch
from torch import autograd
from torch.autograd import Variable 

from layers.select_nd import SelectND

class ConcatWithSelection(autograd.Function):

  def __init__(self, idxs, word_size):
    self.idxs = idxs
    self.word_size = word_size

  def forward(self, with_cat, actor_states):
    self.save_for_backward(with_cat, actor_states)
    res = with_cat.clone()
    res[self.idxs] = actor_states
    return res


  def backward(self, grad_output):
    words, actor_states = self.saved_tensors
    gWords = grad_output
    gActor_states = grad_output[:, self.word_size:][self.idxs].clone()
    return gWords, gActor_states

