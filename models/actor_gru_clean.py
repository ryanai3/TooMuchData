import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as nf
#from .hsm import HSM
from layers.cu_hsm import HSM 
from layers.sentence_embedders import FastMeanWordVectorEmbedder as SentenceEmbedder
from layers.concat_with_select import ConcatWithSelection
from layers.select_nd import SelectND
from layers.detach import InPlaceZeroDetach

#from .actor_pool import FixedCastActorHolder
from layers.actor_pool import FixedSizeActorPool

class ActorGRU(nn.Module):

  def __init__(self, params):
    self.__dict__.update(params)
    super(ActorGRU, self).__init__()
    self.embed = nn.Embedding(
      num_embeddings = self.n_vocab + 3,
      embedding_dim = self.word_dim,
    )
    self.gru = nn.GRUCell(
      input_size = self.word_dim + self.actor_n_units,
      hidden_size = self.n_units
    )
    self.dropout = nn.Dropout(
      p = self.drop_prob,
      inplace=False
    )
    self.bn = nn.BatchNorm1d(
      num_features = self.n_units
    )
    self.hsm = HSM(
      input_size = self.n_units + self.actor_n_units,
      #n_vocab = self.n_vocab
      huff_tree = self.huffman_tree,
    )
    self.sentence_embedder = SentenceEmbedder(
      batch_size = self.batch_size,
      vec_dim = self.word_dim
    )
    self.actor_pool = FixedSizeActorPool( #FixedCastActorHolder(
      input_size = self.word_dim,
      hidden_size = self.actor_n_units,
      cast_size = self.actor_cast_size,
      batch_size = self.batch_size,
      bptt_len = self.actor_bptt_len
    )
    self.query_w = nn.Linear(
      in_features = self.n_units,
      out_features = self.actor_n_units
    )
    self.batch_range = torch.arange(0, self.batch_size).long().cuda()

  def init_params(self):
    self.hsm.init_params()

  def init_state(self, train = True):
    volatile = not train
    self.state = Variable(torch.zeros(self.batch_size, self.n_units).cuda(), volatile = volatile)
    self.zero_out = torch.zeros(self.n_units).cuda()
    self.hsm.init_state(train)
    self.sentence_embedder.init_state(train)
    self.actor_pool.init_state(train)

  def stop_bptt(self):
    self.state.detach_()

  def concat_with_mentioned_actor_states(self, x, actor_locs, actor_ids):
    x = torch.cat([x, Variable(torch.zeros(x.size()[0], self.actor_n_units).cuda(), requires_grad=False)], 1)   
    if actor_locs.dim() > 0:
      lt_idxs = torch.stack([actor_locs, actor_ids[actor_locs]]).t()
      selector = SelectND(lt_idxs)
      mentioned_actor_states = selector(self.actor_pool.state)
      cws = ConcatWithSelection(actor_locs, self.word_dim)
      x = cws(x, mentioned_actor_states)
    return x

  def end_stories(self, story_ends):
    self.state = InPlaceZeroDetach(story_ends)(self.state)
    self.actor_pool.end_stories(story_ends)

  def forward(self, batch):
    wX, wY, am_idX, am_idY, am_locX, am_locY, sam_loc, sam_id, se_loc, sentence_end, story_end = [batch[k].pin_memory().cuda(async=True)
      for k in ['wX', 'wY', 'am_idX', 'am_idY', 'am_locX', 'am_locY', 'sam_loc', 'sam_id', 'se_loc', 'sentence_end', 'story_end']]
    wX = Variable(wX.clamp(0, self.n_vocab + 2))
    wY = Variable(wY.clamp(0, self.n_vocab + 2))

    am_idX = am_idX.clamp(-1, self.actor_cast_size - 1)
    am_idY = am_idY.clamp(-1, self.actor_cast_size - 1)
    wv = self.embed(wX)
    sentence_vecs = self.sentence_embedder(wv, se_loc, sam_loc)
    if sentence_vecs is not None:
      actor_h = self.actor_pool(sentence_vecs, sam_loc, sam_id, story_end)
    wvc = self.concat_with_mentioned_actor_states(wv, am_locX, am_idX)
    h = self.gru(wvc, self.state)
    bn_h = self.bn(h)
    query = nf.tanh(self.query_w(bn_h))
    attentive_read = self.actor_pool.attend(query)
    self.state = h
    if story_end.dim() > 0:
      self.end_stories(story_end)
    late_fuse = self.dropout(torch.cat((bn_h, attentive_read), 1))
    loss = self.hsm(late_fuse, wY.data)/self.batch_size
    return loss
