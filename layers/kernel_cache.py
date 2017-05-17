from os import path

#kernel compilation related imports
from pynvrtc.compiler import Program
from cupy.cuda.function import Module
from cupy.cuda import device

from layers import kernel_dir

class UnCompiledKernel():
  def __init__(self, kernel_name, func_name, device_id):
    self.func_name = func_name or kernel_name
    self.name = kernel_name + ".cu"
    with open(path.join(kernel_dir, self.name), 'r') as cu_f:
      self.kernel_source = cu_f.read().encode()
    self.prog = Program(self.kernel_source, self.name.encode())
  
  def compile_for_device(): #SingleDeviceKernel
    pass
  
class SingleDeviceKernel():
  def __init__(self, kernel_prog, ):
    self.func_name = func_name or kernel_name
    self.name = kernel_name + ".cu"
    with open(path.join(kernel_dir, self.name), 'r') as cu_f:
      self.kernel_source = cu_f.read().encode()
    self.prog = Program(self.kernel_source, self.name.encode())
    ptx = self.prog.compile([self.get_compute_arch_arg(device_id)])
    self.module = Module()
    self.module.load(ptx.encode())

  def prep_args(self, kwargs):
    args = []
    for k, v in kwargs.items():
      try:
        args.append(v.data_ptr())
      except:
        args.append(v)
    return args

  def linear_launch(num_threads, *args):
    kernel_func = self.module.get_function(self.func_name)
    kernel_func.linear_launch(
      num_threads,
      args = self.prep_args(args),
      stream=Stream(
        ptr = torch.cuda.current_stream().cuda_stream
      )
    )   

class KernelCache():
  def __init__(self, kernel_name, func_name=None):
    self.func_name = func_name or kernel_name
    self.name = kernel_name + ".cu"
    with open(path.join(kernel_dir, self.name), 'r') as cu_f:
      self.kernel_source = cu_f.read().encode()
    self.prog = Program(self.kernel_source, self.name.encode()) 
    self.cache = {}

  def cached(self, device_id):
    try:
      kernel_func = self.cache[device_id].get_function(self.func_name)
    except KeyError:
      self.cache[device_id] = self.compile_and_prep_kernel(device_id)
      kernel_func = self.cache[device_id].get_function(self.func_name)
    return kernel_func

  def compile_and_prep_kernel(self, device_id):
    ptx = self.prog.compile([self.get_compute_arch_arg(device_id)])
    module = Module()
    module.load(ptx.encode())
    return module

  def get_compute_arch_arg(self, device_id):
    return "-arch=compute_{0}".format(
      device.Device(device_id).compute_capability\
    ).encode()

  def prep_args(self, kwargs):
    args = []
    for k, v in kwargs.items():
      try:
        args.append(v.data_ptr())
      except:
        args.append(v)
    return args


