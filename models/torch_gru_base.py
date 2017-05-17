import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as nf

#from .hsm import HSM
from layers.cu_hsm import HSM

class FakeGRU(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(FakeGRU, self).__init__()
    self.W_r = nn.Linear(input_size, hidden_size)
    self.U_r = nn.Linear(hidden_size, hidden_size)
    self.W_z = nn.Linear(input_size, hidden_size)
    self.U_z = nn.Linear(hidden_size, hidden_size)
    self.W = nn.Linear(input_size, hidden_size)
    self.U = nn.Linear(hidden_size, hidden_size)

  def forward(self, x, h):
    r = nf.sigmoid(self.W_r(x) + self.U_r(h))
    z = nf.sigmoid(self.W_z(x) + self.U_z(h))
    h_bar = nf.tanh(self.W(x) + self.U(r * h))
    h_new = (z * h_bar) + (1 - z) * h
    return h_new

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


  def init_params(self):
    self.hsm.init_params()

  def stop_bptt(self):
    self.state.detach_()
    #self.state = self.state.detach()

  def init_state(self, train = True):
    volatile = not train
    self.state = Variable(torch.zeros(self.batch_size, self.n_units).cuda(), volatile = volatile)
    self.zero_out = torch.zeros(self.n_units).cuda()

  def __call__(self, xx, train = True):
    word_next_tup, stop_idxs = xx
    for idx in stop_idxs:
      self.state[idx].detach_()
      self.state.data[idx] = self.zero_out
#    x, t = [np.clip(l, 0, self.n_vocab + 2) for l in word_next_tup]
    x = Variable(
      torch.from_numpy(
        np.clip(word_next_tup[0], 0, self.n_vocab + 2).astype(np.int64)
      ).pin_memory().cuda(async=True),
      volatile = not train
    )
    t = torch.from_numpy(
        np.clip(word_next_tup[1], 0, self.n_vocab + 2)
    ).pin_memory().cuda(async=True)

#    x, t = [torch.from_numpy(np.clip(l, 0, self.n_vocab + 2)).long().cuda()
#        for l in word_next_tup]
#    if not train:
#    x = Variable(torch.from_numpy(x).long().cuda(), volatile = not train)
#    t = Variable(t, volatile = not train)
#    t = t.long().cuda()
#    t = Variable(t, volatile = not train)
    x0 = self.embed(x)
    h = self.gru(x0, self.state)
    self.state = h
    loss = self.hsm(h, t)/self.batch_size
    return loss
