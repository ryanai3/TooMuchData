from collections import defaultdict, namedtuple

import numpy as np

#torch imports
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Function, Variable

#kernel compilation related imports
from pynvrtc.compiler import Program
from cupy.cuda.function import Module
from cupy.cuda import device

from string import Template

#needed for huffmantree stuff:
from chainer.links.loss.hierarchical_softmax import TreeParser

Stream = namedtuple('Stream', ['ptr'])

class HSM(nn.Module):
  def __init__(self, input_size, huff_tree):
    super(HSM, self).__init__()
    self.input_size = input_size
    self.tp = TreeParser()
    self.tp.parse(huff_tree)
    self.n_decisions = self.tp.size()
    self.W = nn.Parameter(
      torch.Tensor(self.n_decisions, self.input_size)
    )
    paths_d = self.tp.get_paths()
    codes_d = self.tp.get_codes()
    self.n_vocab = max(paths_d.keys()) + 1

    paths_a = [paths_d[i] for i in range(self.n_vocab)]
    codes_a = [codes_d[i] for i in range(self.n_vocab)]
    self.register_buffer(
      'paths', 
      torch.cat([torch.from_numpy(path) for path in paths_a])
    )
    self.register_buffer(
      'codes',
      torch.cat([torch.from_numpy(code) for code in codes_a])
    )
    lens_a = [len(path) for path in paths_a]
    self.register_buffer('lens', torch.IntTensor(lens_a))
    self.register_buffer(
      'begins', 
      torch.IntTensor([0] + lens_a).cumsum(0)
    )
    self.forward_kernel_cache = HSMForwardKernelCache()
    self.backward_kernel_cache = HSMBackwardKernelCache()
#    self.forward_backward_kernel_cache = HSMForwardBackwardKernelCache()


  def init_params(self):
    #init.kaiming_normal(self.W)
    init.orthogonal(self.W)

  def init_state(self, train=True):
    self.W._grad = Variable(torch.zeros(self.W.size()).cuda())

  def forward(self, x, t): # returns a Function, with x and t applied
    loss_func = HSMLoss(
      W = self.W.data,
      t = t,
      paths = self.paths,
      codes = self.codes,
      begins = self.begins,
      lens = self.lens, 
      gW = self.W.grad,
      f_kern_cache = self.forward_kernel_cache,
      b_kern_cache = self.backward_kernel_cache
#      fb_kern_cache = self.forward_backward_kernel_cache
    )
#    import pdb; pdb.set_trace()
#    from torch.autograd import gradcheck
#    test = gradcheck(loss_func, (Variable(x.data, requires_grad=True), self.W), eps=1e-2, atol=1e-4)
    return loss_func(x)# , self.W)

class NewHSMLoss(Function):
  def __init__(self, t, paths, codes, begins, lens, gW, fb_kern_cache):
    self.t = t
    self.n_ex = self.t.size()[0]
    self.paths = paths
    self.codes = codes
    self.begins = begins
    self.lens = lens
    self.gW = gW
    self.forward_backward_kernel = fb_kern_cache

  def forward(self, x, W):
#    import pdb; pdb.set_trace()
    self.x = x
    self.W = W
    self.save_for_backward(x, W)
    self.max_len = self.lens[self.t.long()].max()
    self.n_in = x.size()[1]
    total_len = self.max_len * x.size()[0]
    self.num_threads = total_len
    self.ls = torch.zeros(total_len,).cuda().contiguous()
    self.gx = torch.zeros(self.x.size()).cuda()#.contiguous()
#    self.gW = torch.zeros(self.W.size()).cuda()#.contiguous()
#    import pdb; pdb.set_trace()
    self.forward_backward_kernel(
      num_threads = total_len,
      x = self.x,
      W = self.W,
      t = self.t,
      paths = self.paths,
      codes = self.codes,
      begins = self.begins,
#      lens = self.lens,
      n_in = self.n_in,
      max_len = self.max_len,
      n_ex = self.n_ex,
      ls = self.ls,
      gx = self.gx,
      gW = self.gW
    )
    return self.ls.sum(0)

  def backward(self, grad_output):
#    multiplier = grad_output[0]
#    scaled_gx = multiplier * self.gx
#    scaled_gW = multiplier * self.gW
#    return scaled_gx, scaled_gW
    return self.gx, self.gW
  

class HSMLoss(Function):

  def __init__(self, W, t, paths, codes, begins, lens, gW, f_kern_cache, b_kern_cache):
    self.W = W
    self.t = t
    self.n_ex = self.t.size()[0]
    self.paths = paths
    self.codes = codes
    self.begins = begins
    self.lens = lens
    self.gW = gW
    self.forward_kernel = f_kern_cache
    self.backward_kernel = b_kern_cache

  def forward(self, x):
    self.x = x
    self.save_for_backward(x)
    self.max_len = self.lens[self.t.long()].max()
    self.n_in = x.size()[1]
    total_len = self.max_len * x.size()[0]
    self.num_threads = total_len
    self.ls = torch.zeros(total_len,).cuda().contiguous()
    self.wxy = torch.zeros(total_len,).cuda().contiguous()
#    import pdb; pdb.set_trace()
    self.forward_kernel(
      num_threads = total_len,
      x = self.x,
      W = self.W,
      t = self.t,
      paths = self.paths,
      codes = self.codes,
      begins = self.begins,
#      lens = self.lens,
      n_in = self.n_in,
      max_len = self.max_len,
      n_ex = self.n_ex,
      ls = self.ls,
      wxy = self.wxy
    )
    return self.ls.sum(0)

  def backward(self, grad_output):
    gx = torch.zeros(self.x.size()).cuda().contiguous()
    self.backward_kernel(
      num_threads=self.num_threads,
      wxy = self.wxy,
      x = self.x,
      W = self.W,
      t = self.t,
      paths = self.paths,
      codes = self.codes,
      begins = self.begins,
#      lens = self.lens,
      gLoss = grad_output,
      n_in = self.n_in,
      max_len = self.max_len,
      n_ex = self.n_ex,
      gx = gx,
      gW = self.gW.data
#      gW = self.W.grad
    )
#    print(gx)
#    print(gW)
#    import pdb; pdb.set_trace()
    return gx#, self.W.grad
  
from .kernel_cache import KernelCache

class HSMForwardBackwardKernelCache(KernelCache):
  def __init__(self):
    super(HSMForwardBackwardKernelCache, self).__init__('bhsm_forward_backward')
          
  def __call__(self, num_threads, **kwargs):
    kernel_func = self.cached(kwargs['x'].get_device())
    args = self.prep_args(kwargs)
    kernel_func.linear_launch(
      num_threads,
#      grid=(min(num_threads // 128 + 1, 65536), 1, 1),
#      block=(min(num_threads, 128), 1, 1),
      args=args,
      stream=Stream(
        ptr=torch.cuda.current_stream().cuda_stream
      )
    )

class HSMForwardKernelCache(KernelCache):
  def __init__(self):
    super(HSMForwardKernelCache, self).__init__('bhsm_forward')
          
  def __call__(self, num_threads, **kwargs):
    kernel_func = self.cached(kwargs['x'].get_device())
    args = self.prep_args(kwargs)
    kernel_func.linear_launch(
      num_threads,
#      grid=(min(num_threads // 128 + 1, 65536), 1, 1),
#      block=(min(num_threads, 128), 1, 1),
      args=args,
      stream=Stream(
        ptr=torch.cuda.current_stream().cuda_stream
      )
    )

class HSMBackwardKernelCache(KernelCache):
  def __init__(self):
    super(HSMBackwardKernelCache, self).__init__('bhsm_backward')

  def __call__(self, num_threads, **kwargs):
    kernel_func = self.cached(kwargs['x'].get_device())
    args = self.prep_args(kwargs)
#    s_x= 1024
#    s_y= 1024
#    mx = 65536
    kernel_func.linear_launch(
      num_threads,
#      grid=(min(n_x // s_x + 1, mx), min(n_y // s_y + 1, mx), 1),
#      block=(min(n_x, s_x), min(n_y, s_y, 1), 1),
      args=args,
      stream=Stream(
        ptr=torch.cuda.current_stream().cuda_stream
      )
    )
