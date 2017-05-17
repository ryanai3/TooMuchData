import numpy as np

import cupy
import chainer
from chainer import initializers
from chainer import Variable

import chainer.links as L
import chainer.functions as F

class GRU_Baseline(chainer.Chain):

  def __init__(self, params):
    self.__dict__.update(params)
    super(GRU_Baseline, self).__init__(
      embed = L.EmbedID(self.n_vocab + 3, self.word_dim),
      gru = L.GRU(
        n_units = self.n_units,
        n_inputs = self.word_dim
      ),
#      l2v = L.Linear(
#        in_size = self.n_units,
#        out_size = self.n_vocab + 2
#      )
#      blackout = L.BlackOut(
#        in_size = self.n_units,
#        counts = self.vocab_counts,
#        sample_size = self.blackout_sample_size
#      )
#      bottleneck = L.Linear(
#        in_size = self.n_units,
#        out_size = 128
#      ),
      bh_softmax = L.BinaryHierarchicalSoftmax(
        in_size = self.n_units,
        tree = self.huffman_tree
      )
    )
#    chainer.init_weight(self.blackout.W.data, chainer.initializers.HeNormal())

  def stop_bptt(self):
    self.state.unchain_backward()

  def init_state(self, train = True):
    flag = 'OFF' if train else 'ON'
    self.state = Variable(cupy.zeros((self.batch_size, self.n_units), dtype=cupy.float32), flag)
    self.zero_out = cupy.zeros((self.n_units,))

  def __call__(self, xx, train = True):
    word_next_tup, stop_idxs = xx
    for idx in stop_idxs:
      self.state[idx].unchain_backward()
      self.state.data[idx] = self.zero_out
    x, t = [cupy.array(np.clip(l, 0, self.n_vocab + 2),dtype=cupy.int32) for l in word_next_tup]
    if not train:
      x = Variable(x, volatile='ON')
      t = Variable(t, volatile='ON')
    x0 = self.embed(x)
    h = self.gru(self.state, x0)
    self.state = h
#    h = F.relu(self.bottleneck(h))
#    h = self.bottleneck(h)
#    v = self.l2v(h)
#    loss = F.softmax_cross_entropy(v, t)
#    if train:
#      loss = self.blackout(h, t)
#    else:
#      v = F.linear(h, self.blackout.W)
#      loss = F.softmax_cross_entropy(v, t)
#    import pdb; pdb.set_trace()
    loss = self.bh_softmax(h, t) / self.batch_size
#    chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
    return loss
