import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as nf
#from .hsm import HSM
from layers.cu_hsm import HSM 
#from layers.sentence_embedders import MeanWordVectorEmbedder as SentenceEmbedder
from layers.sentence_embedders import PresetRandomVectorEmbedder as SentenceEmbedder
from layers.concat_with_select import ConcatWithSelection
from layers.select_nd import SelectND

from .actor_holder import FixedCastActorHolder

class ActorGRU(nn.Module):

  def __init__(self, params):
    self.__dict__.update(params)
    super(ActorGRU, self).__init__()
    self.embed = nn.Embedding(
      num_embeddings = self.n_vocab + 3,
      embedding_dim = self.word_dim,
    )
    self.gru = nn.GRUCell(
      input_size = self.word_dim,# + self.actor_n_units,
      hidden_size = self.n_units
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
    self.actor_holder = FixedCastActorHolder(
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


  def init_params(self):
    self.hsm.init_params()

  def stop_bptt(self):
    self.state.detach_()
    #self.state = self.state.detach()

  def init_state(self, train = True):
    volatile = not train
    self.state = Variable(torch.zeros(self.batch_size, self.n_units).cuda(), volatile = volatile)
    self.zero_out = torch.zeros(self.n_units).cuda()
    self.sentence_embedder.init_state(train)
    self.actor_holder.init_state(train)

  def __call__(self, batch, train = True):
#    (x, am_bix, am_idx), (t, am_biy, am_idy), (sam_batch_i, asms), (sentence_stop_idxs, story_stop_idxs) = batch
    x, t, am_idx, am_idy, asms, sentence_end, story_end = [batch[k] for k in ['wX', 'wY', 'am_idx', 'am_idY', 's_a_id', 'sentence_end', 'story_end', 'id']]
    for idx in story_stop_idxs:
      self.state[idx].detach_()
      self.state.data[idx] = self.zero_out
    x = Variable(
      torch.from_numpy(
        np.clip(x, 0, self.n_vocab + 2).astype(np.int64)
      ).pin_memory().cuda(async=True),
      volatile = not train
    )
    t = torch.from_numpy(
        np.clip(t, 0, self.n_vocab + 2)
    ).pin_memory().cuda(async=True)
    sam_batch_i = torch.cuda.LongTensor(sam_batch_i)
    asms = torch.cuda.LongTensor(asms)
    story_stop_idxs = torch.cuda.LongTensor(story_stop_idxs)


    x0 = self.embed(x)
    sentence_vecs = self.sentence_embedder(x0, sentence_stop_idxs, sam_batch_i)
    actor_h = self.actor_holder(sentence_vecs, sam_batch_i, asms, story_stop_idxs)
#    x0 = torch.cat([x0, Variable(torch.zeros(x.size()[0], self.actor_n_units).cuda(), requires_grad=False)], 1)
#    if asms.dim() > 0:
#      lt_idxs = torch.stack([sam_batch_i, asms]).t()
#      mentioned_actor_states = SelectND(lt_idxs)(self.actor_holder.state)
#      cws = ConcatWithSelection(sam_batch_i, self.word_dim)
#      x0 = cws(x0, mentioned_actor_states)
#      x0 = ConcatWithSelection(sam_batch_i, self.word_dim)(x0, mentioned_actor_states)
    h = self.gru(x0, self.state)
    query = nf.tanh(self.query_w(h))
    attentive_read = self.actor_holder.attend(query)
    self.state = h
    late_fuse = torch.cat((h, attentive_read), 1)
    loss = self.hsm(late_fuse, t)/self.batch_size
    return loss
