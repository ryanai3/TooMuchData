import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as nf

#from .hsm import HSM
from .cu_hsm import HSM

class SentenceEmbedder():
  def __init__(self, batch_size, vec_dim):
    self.batch_size = batch_size
    self.vec_dim = vec_dim

  def init_state(self, train=True):
    volatile = not train
#    import pdb; pdb.set_trace()
    self.state = Variable(
      torch.zeros((self.batch_size, self.vec_dim)).cuda(),
      volatile = volatile
    )
    self.zero_out = torch.zeros((self.vec_dim,)).cuda()
    self.lens = torch.zeros((self.batch_size,)).cuda()

  def take_words(self, word_vecs):
    self.state += word_vecs
    self.lens += 1

  def finish_sentences(self, idxs):
    for idx in idxs:
      self.state[idx].detach_()
      self.state[idx] = self.zero_out
      self.lens[idx] = 0

  def collect_sentences(self, idxs):
    try:
      res = torch.stack([self.state[idx] for idx in idxs])
    except:
      import pdb; pdb.set_trace()
      print(32)
    return res

class GRU_Baseline(nn.Module):

  def __init__(self, params):
    self.__dict__.update(params)
    super(GRU_Baseline, self).__init__()
    self.embed = nn.Embedding(
      num_embeddings = self.n_vocab + 3,
      embedding_dim = self.word_dim,
    )
    self.gru = nn.GRUCell(
      input_size = self.word_dim,
      hidden_size = self.n_units
    )
    self.hsm = HSM(
      input_size = self.n_units,
      #n_vocab = self.n_vocab
      huff_tree = self.huffman_tree,
    )
    self.sentence_embedder = SentenceEmbedder(
      batch_size = self.batch_size,
      vec_dim = self.word_dim
    )
    self.actor_gru = nn.GRUCell(
      input_size = self.n_units,
      hidden_size = 32
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
    self.actor_state = Variable(
      torch.zeros(self.batch_size, self.n_actors, self.n_units)
    )
    self.actor_zero_out = torch.zeros(self.n_actors, self.n_units)

  def __call__(self, batch, train = True):
    (x, am_bix, am_idx), (t, am_biy, am_idy), (sam_batch_i, asms), (sentence_stop_idxs, story_stop_idxs) = batch
    for idx in story_stop_idxs:
      self.state[idx].detach_()
      self.state.data[idx] = self.zero_out
      self.actor_state[idx].detach_()
      self.actor_state[idx] = self.actor_zero_out
#    x, t = [np.clip(l, 0, self.n_vocab + 2) for l in word_next_tup]
#    print(x)
#    print(sentence_stop_idxs)
    x = Variable(
      torch.from_numpy(
        np.clip(x, 0, self.n_vocab + 2).astype(np.int64)
      ).pin_memory().cuda(async=True),
      volatile = not train
    )
    t = torch.from_numpy(
        np.clip(t, 0, self.n_vocab + 2)
    ).pin_memory().cuda(async=True)
    asms = np.clip(np.array(asms), 0, self.n_actors - 1)
    x0 = self.embed(x)
    if sam_batch_i:
      sentence_vecs = self.sentence_embedder.collect_sentences(sam_batch_i)
      mentioned_actor_states = torch.stack([self.actor_state[bi, ai] for bi, ai in zip(sam_batch_i, asms)])
    if sentence_stop_idxs:
      self.sentence_embedder.finish_sentences(sentence_stop_idxs)   
    self.sentence_embedder.take_words(x0)
    if sam_batch_i:
      import pdb; pdb.set_trace() 
      next_mentioned_actor_states = self.actor_gru(sentence_vecs, mentioned_actor_states)
      for bi, ai, v in zip(sam_batch_i, asms, next_mentioned_actor_states):
        self.actor_state[bi, ai] = v

    h = self.gru(x0, self.state)
    self.state = h
    loss = self.hsm(h, t)/self.batch_size
    return loss
