import torch
from torch import nn
import torch.nn.functional as nf
from torch.nn import init
from torch.autograd import Variable

import numpy as np

from chainer.links.loss.hierarchical_softmax import TreeParser

#class HSM(nn.Module):
#  def __init__(self, input_size, vocab_size):

class HSBad(nn.Module):
  def __init__(self, input_size, n_vocab):
    super(HSM, self).__init__()
    self.input_size = input_size
    self.n_vocab = n_vocab + 3
    self.l2v = nn.Linear(
      in_features = self.input_size,
      out_features = self.n_vocab
    )
  
  def init_params(self):
    pass

  def __call__(self, x, t):
    v = self.l2v(x)
    loss = nf.cross_entropy(v, t)
    return loss


class HSFil(nn.Module):
  def __init__(self, input_size, huff_tree):
    super(HSM, self).__init__()
    self.input_size = input_size
    self.tp = TreeParser()
    self.tp.parse(huff_tree)
    self.n_decisions = self.tp.size()
#    self.decs = nn.Embedding(
#      num_embeddings = self.n_decisions,
#      embedding_dim = self.input_size
#    )
    self.decs = nn.Parameter(
      torch.Tensor(self.n_decisions, self.input_size)
    )
    paths_d = self.tp.get_paths()
    codes_d = self.tp.get_codes()
    self.n_vocab = max(paths_d.keys()) + 1
    self.paths = [paths_d[i] for i in range(self.n_vocab)]
    self.codes = [torch.from_numpy(codes_d[i]) for i in range(self.n_vocab)]
    self.lens = [len(path) for path in self.paths]
    self.begins = np.cumsum([0.0] + self.lens[:-1])

  def init_params(self):
    g_paths = [torch.from_numpy(path).cuda().long() for path in self.paths]
#    g_paths = torch.from_numpy(np.concatenate(self.paths)).cuda().long()
    self.path_mats = [self.decs[g_path].cpu() for g_path in g_paths]
  def __call__(self, x, t):
#    import pdb; pdb.set_trace()
    curr_path_mats = [self.path_mats[i] for i in t]
    vv = [pm.cuda().mv(x_i) for pm, x_i in zip(curr_path_mats, x)]
    loss = -nf.logsigmoid(torch.cat(vv)).sum()/x.size()[0]
#    r_paths = []
#    r_codes = []
#    r_xs = []
#    paths = torch.cat([self.paths[t_i] for t_i in t])
#    codes = torch.cat([self.codes[t_i] for t_i in t])
#    import pdb; pdb.set_trace()
#    for x_i, t_i in zip(x, t):
#      path, code = self.paths[t_i], self.codes[t_i]
#      r_paths.append(self.paths[t_i])
#      r_codes.append(self.codes[t_i])
#      r_xs.append(x_i.repeat(len(path), 1))
##      r_xs.append(x_i.expand(len(path), self.input_size)) 
#    #g_paths = Variable(torch.from_numpy(np.concatenate(r_paths)).long().cuda(), requires_grad=False)
##    g_codes = Variable(torch.from_numpy(np.concatenate(r_codes)).cuda(), requires_grad=False)
#    g_xs = torch.cat(r_xs)
##    loss = nf.binary_cross_entropy(self.decs(g_paths) * g_xs, g_codes, size_average=False)
#    loss = nf.logsigmoid((self.decs(g_paths) * g_xs).sum(1) * g_codes).sum()/x.size()[0]
    return loss
 
class HSM(nn.Module):
  def __init__(self, input_size, huff_tree):
    super(HSM, self).__init__()
    self.input_size = input_size
    self.tp = TreeParser()
    self.tp.parse(huff_tree)
    self.n_decisions = self.tp.size()
#    self.decs = nn.Linear(
#      in_features = self.input_size,
#      out_features = self.n_decisions
#    )
#    self.decs = nn.Parameter(
#      torch.Tensor(self.n_decisions, self.input_size)
#    )
    self.decs = nn.Embedding(
      num_embeddings = self.n_decisions,
      embedding_dim = self.input_size
    )
    self.paths_d = self.tp.get_paths()
    self.codes_d = self.tp.get_codes()
    self.max_path = max([len(v) for v in self.paths_d.values()])
    self.max_code = max([len(v) for v in self.codes_d.values()])
    self.n_vocab = max(self.paths_d.keys()) + 1

  def init_params(self):
#    init.kaiming_normal(self.decs)
    self.paths = [self.paths_d[i] for i in range(self.n_vocab)]
    self.paths = [np.pad(path, (0, max(0, self.max_path - len(path))), mode='constant') for path in self.paths]
    self.paths = torch.stack([torch.from_numpy(path) for path in self.paths], 0).long().cuda()

    self.codes = [self.codes_d[i] for i in range(self.n_vocab)]
    self.codes = [np.pad(code, (0, max(0, self.max_code - len(code))), mode='constant') for code in self.codes]
    self.codes = torch.stack([torch.from_numpy(code) for code in self.codes], 0).cuda()

  def __call__(self, x, t):
#    import pdb; pdb.set_trace()
    g_t = torch.from_numpy(t).cuda().long()
    ws = self.decs(Variable(self.paths[g_t]))
#    ws = self.decs(self.paths[t].view(-1)).view(x.size()[0], self.max_path, self.input_size)
    scores = ws.bmm(x.unsqueeze(2)).squeeze() * Variable(self.codes[g_t])
    nz_mask = scores.ne(0).detach()
    loss = -nf.logsigmoid(scores.masked_select(nz_mask)).sum()/x.size()[0]
#    ws = [self.decs[self.paths[t_i]] * self.codes[t_i] for t_i in t]
#    ws = torch.cat([self.decs[self.paths[t_i]].mv(x_i) for t_i, x_i in zip(t, x)])
#    cs = Variable(torch.cat([self.codes[t_i] for t_i in t]))
#    loss = -nf.logsigmoid(ws * cs).sum()/x.size()[0]
    return loss
  

class HSFail(nn.Module):
  def __init__(self, input_size, vocab_size, branch_factor):
    super(HSM, self).__init__()
    self.input_size = input_size
    self.vocab_size = vocab_size
    self.branch_factor = branch_factor
    self.level1 = nn.Linear(
      in_features = self.input_size,
      out_features = self.branch_factor
    )
    self.level2_w = nn.Parameter(
      torch.Tensor(self.branch_factor, self.branch_factor, self.input_size)
    )
#    self.level2_b = nn.Parameter(
#      torch.Tensor(self.branch_factor, self.branch_factor)
#    )

  def init_params(self):
    init.kaiming_normal(self.level2_w)
#    init.kaiming_normal(self.level2_b)
    pass

  def forward(self, x, t):
#    import pdb; pdb.set_trace()
    t1 = (t / self.branch_factor).long().cuda()
    t2 = (t % self.branch_factor)
    l1 = self.level1(x)
    
    l1_ce = nf.cross_entropy(l1, Variable(t1))
#    l1_log_softmax = sum([res[idx] for res, idx in zip(nf.log_softmax(l1), t1)])
#    l1_log_softmax = nf.log_softmax(l1).t()[t1].diag()
###    l2_w = [self.level2_w[idx] for idx in t2]
#    l2_w = self.level2_w[t1]
##    l2_b = self.level2_b[t1]
###    l2_aff = torch.stack([mat.mv(vec) for mat, vec in zip(l2_w, x)], 0)# + l2_b
    l2_aff = torch.stack([self.level2_w[idx].mv(vec) for idx, vec in zip(t2, x)], 0)
    l2_ce = nf.cross_entropy(l2_aff, Variable(t2.cuda()).long())
#    l2_aff = [mat.addmv(l2_b, vec) for mat, vec in zip(l2_w, x)]
#    l2_aff = l2_w.bmm(x.unsqueeze(2)).squeeze() + l2_b
##    l2_log_softmax = torch.stack([res[idx] for res, idx in zip(nf.log_softmax(l2_aff), t2)], 0)
#    l2_log_softmax = nf.log_softmax(l2_aff).t()[t2].diag()
##    ce = - (l1_log_softmax + l2_log_softmax).sum()/x.size()[0]
#    ce = -l1_log_softmax/x.size()[0]
    return l1_ce + l2_ce
