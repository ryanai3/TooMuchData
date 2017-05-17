import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as nf

class SentenceEmbedder(nn.Module):
  def __init__(self, batch_size, vec_dim):
    raise NotImplementedError

  def init_state(self, train=True):
    raise NotImplementedError

  def __call__(self, word_vecs, stop_idxs):
    raise NotImplementedError

class ViewWholeSentenceEmbedder(SentenceEmbedder):
  def __init__(self, batch_size, vec_dim):
    self.batch_size = batch_size
    self.vec_dim = vec_dim

  def init_state(self, train=True):
    self.sentences = [[] for i in range(self.batch_size)]

  def embed_whole_sentence(self, sentence):
    raise NotImplementedError

  def __call__(self, word_vecs, stop_idxs, collect_idxs):
    stacked = []
    for idx in collect_idxs:
      stacked.append(torch.stack(self.sentences[idx]))
    for idx in stop_idxs:
      self.sentences[idx] = []
    for sentence, word in zip(self.sentences, word_vecs):
      sentence.append(word)
    embeddings = None
    try:
      embeddings = torch.cat( 
        [self.embed_whole_sentence(sentence) for sentence in stacked],
        0
      )
    except:
      pass
    return embeddings

class MeanWordVectorEmbedder(ViewWholeSentenceEmbedder):
  def embed_whole_sentence(self, sentence):
    return sentence.mean(0)

class PresetRandomVectorEmbedder():
  def __init__(self, batch_size, vec_dim):
    self.batch_size = batch_size
    self.vec_dim = vec_dim
    self.state = Variable(torch.randn(batch_size, vec_dim).cuda(), requires_grad=False)

  def init_state(self, train=True):
    pass

  def take_words(self, word_vecs):
    pass

  def __call__(self, word_vecs, stop_idxs, collect_idxs):
    if len(collect_idxs) > 0:
      return self.state[:len(collect_idxs)]

class FastMeanWordVectorEmbedder():
  def __init__(self, batch_size, vec_dim):
    self.batch_size = batch_size
    self.vec_dim = vec_dim

  def init_state(self, train=True):
    self.state = Variable(
      torch.zeros((self.batch_size, self.vec_dim)).cuda(),
      requires_grad=False
    )
    self.lens = Variable(
      torch.zeros((self.batch_size, 1)).cuda(),
      requires_grad=False
    )

  def zero_out(self, idxs):
    self.state.data[idxs] = 0
    self.lens.data[idxs] = 0

  def __call__(self, word_vecs, stop_idxs, collect_idxs):
    res = None
    if collect_idxs.dim() > 0:
      num = self.state[collect_idxs]
      den = self.lens[collect_idxs]
      res = num/den.expand_as(num)
      self.zero_out(collect_idxs)
    if stop_idxs.dim() > 0:
      self.zero_out(stop_idxs)
    self.state = word_vecs + self.state
    self.lens.data += 1
    return res

class LazyMeanWordVectorEmbedder():
  def __init__(self, batch_size, vec_dim):
    self.batch_size = batch_size
    self.vec_dim = vec_dim

  def init_state(self, train=True):
    volatile = not train
    self.sentences = []
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
 
