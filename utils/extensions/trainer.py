import torch

from copy import deepcopy

from extensions import Extension

class TQDMSourceUpdater(Extension):

  def __init__(self, keys=None):
    self.keys = keys

  def __call__(self, recipe):
    if keys == None:
      self.keys = 
    to_report = {k: recipe.curr_obs[key] for key in keys}
    tqdm = recipe.source
    tqdm.set_postfix(**to_report)
