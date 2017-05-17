import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from layers.select_nd import SelectND, SelectiveUpdate
from layers.detach import InPlaceZeroDetach

class ActorPool(nn.Module):
  def __init__(self, **params):
    self.__dict__.update(params)

    super(ActorPool, self).__init__()
    self.gru = nn.GRUCell(
      input_size = self.input_size,
      hidden_size = self.hidden_size
    )
    try:
      self.attn_query = nn.Linear(
        in_features = self.query_size,
        out_features = self.hidden_size
      )
      self.mention_query = nn.Linear(
        in_features = self.query_size,
        out_features = self.hidden_size
      )
    except AttributeError:
      pass

  def init_state(self, train = True):
    self.count = 0

  def stop_bptt(self):
    self.state.detach_()

  def end_stories(self, story_ends):
    self.state = InPlaceZeroDetach(story_ends)(self.state)

  def convert_idxs(self, batch_idxs, actor_ids):
    raise NotImplementedError

  def get_actors(self, batch_idxs, actor_ids):
    return self._get_actors(self.convert_idxs(batch_idxs, actor_ids))

  def _get_actors(self, idxs):
    raise NotImplementedError

  def update_actors(self, batch_idxs, actor_ids, obs):
    self._update_actors(self.convert_idxs(batch_idxs, actor_ids), obs)

  def _update_actors(self, idxs, obs):
    raise NotImplementedError
   
  def __call__(self, x, batch_idxs, actor_ids, story_stop_idxs):
    actor_ids = actor_ids.clamp(0, self.cast_size - 1)
    idxs = self.convert_idxs(batch_idxs, actor_ids)
    if batch_idxs.dim() > 0:
      selected = self._get_actors(idxs)
      self.count += 1
      new_selected = self.gru(x, selected)
      self._update_actors(idxs, new_selected)
      if story_stop_idxs.dim() > 0:
        self.state = InPlaceZeroDetach(story_stop_idxs)(self.state)
      if self.count % self.bptt_len == 0:
        self.state = self.state.detach()
      return new_selected

  def attn_weights(self, query_vectors):
    raise NotImplementedError

  def attend(self, query_vectors):
    raise NotImplementedError

  def argmax(self, query_vectors):
    raise NotImplementedError

class FixedSizeActorPool(ActorPool):

  def __init__(self, **params):
    super(FixedSizeActorPool, self).__init__(**params)
    

  def init_state(self, train = True):
    volatile = not train
    self.state = Variable(
      torch.zeros(self.batch_size, self.cast_size, self.hidden_size).cuda(),
      volatile = volatile
    )
    super(FixedSizeActorPool, self).init_state(train)

  def convert_idxs(self, batch_idxs, actor_ids):
    return torch.stack([batch_idxs, actor_ids[batch_idxs]]).t()

  def _get_actors(self, lt_idxs):
    return SelectND(lt_idxs)(self.state)

  def _update_actors(self, lt_idxs, new_state):
    new_state = SelectiveUpdate(lt_idxs)(self.state, new_state)
  
  def attn_weights(self, query3d):
    dots = self.state.bmm(query3d).squeeze()
    weights = F.softmax(F.threshold(dots, 0.0001, -1000.0))
    return weights  

  def query(self, h):
    both_qs = torch.stack(
      [F.tanh(self.attn_query(h)), F.tanh(self.mention_query(h))],
      dim = 2
    )
    both_dots = self.state.bmm(both_qs)
    mention_scores = both_dots[:, :, 1]
    attn_dist = F.softmax(both_dots[:, :, 1]).unsqueeze(2)
    attn_read = self.state.transpose(1, 2).bmm(attn_dist).squeeze()    

    return (attn_read, mention_scores)

  def attend(self, query_vectors):
    dots = self.state.bmm(query_vectors.unsqueeze(2)).squeeze()
    weights = F.softmax(F.threshold(dots, 0.0001, -1000.0))
    attended = weights.unsqueeze(1).bmm(self.state).squeeze()
    return attended

  def argmax(self, query_vectors):
    dots = self.state.bmm(query_vectors.unsqueeze(2)).squeeze()
    _, idxs = dots.max(1)
    idxs = idxs.data


