import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from layers.select_nd import SelectND, SelectiveUpdate
from layers.detach import InPlaceZeroDetach

class FixedCastActorHolder(nn.Module):

  def __init__(self, input_size, hidden_size, cast_size, batch_size, bptt_len):
    super(FixedCastActorHolder, self).__init__()
    self.n_units = hidden_size
    self.gru = nn.GRUCell(
      input_size = input_size,
      hidden_size = self.n_units
    )
    self.cast_size = cast_size
    self.batch_size = batch_size
    self.bptt_len = bptt_len
    self.counts = torch.zeros(batch_size, self.cast_size).int()
    self.count = 0

  def init_state(self, train = True):
    volatile = not train
    self.state = Variable(
      torch.zeros(self.batch_size, self.cast_size, self.n_units).cuda(),
      volatile = volatile
    )

  def __call__(self, x, batch_idxs, actor_ids, story_stop_idxs):
    self.count += 1
    actor_ids.clamp_(0, self.cast_size - 1)
    new_h = None
    if batch_idxs.dim() > 0:
      lt_idxs = torch.stack([batch_idxs, actor_ids[batch_idxs]]).t()
      select_read = SelectND(lt_idxs)
      select_update = SelectiveUpdate(lt_idxs)
      selected = select_read(self.state)
      new_selected = self.gru(x, selected)
      new_state = select_update(self.state, new_selected)
      if story_stop_idxs.dim() > 0:
        self.state = InPlaceZeroDetach(story_stop_idxs)(self.state)
      if self.count % 10 == 0:
        self.state = new_state.detach()
      return new_selected

  def attend(self, query_vectors):
    dots = self.state.bmm(query_vectors.unsqueeze(2)).squeeze()
    weights = F.softmax(F.threshold(dots, 0.0001, -1000.0))
    attended = weights.unsqueeze(1).bmm(self.state).squeeze()
    return attended

  def argmax(self, query_vectors):
    dots = self.state.bmm(query_vectors.unsqueeze(2)).squeeze()
    _, idxs = dots.max(1)
    idxs = idxs.data


