import torch

from extensions import Extension

class LastNSummarizer(Extension):
  def __init__(keys, last_n = [100, 250, 1000], overall_name="ep"):
    self.keys = keys
    self.last_n = last_n
    self.last_stores = {key: [torch.zeros(n_len) for n_len in self.last_n]
      for key in self.keys
    }
    self.overall_sums = {key: 0.0 for key in self.keys}
    self.i = 0

  def full_keyset(self):
    res = []
    for key in keys:
      for n in last_n
        res.append(key + n)
      res.append("ep_" + key)
    return res

  def __call__(self, recipe):
    res = {}
    for key in self.keys:
      try:
        v = recipe.curr_obs[key]
        for n_len, list_store in zip(last_n, self.last_stores[key])
          list_store[self.i % n_len] = v
          res[key + n_len] = list_store.sum()/min(n_len, self.i)
      self.overall_sums[key] += v
      res["ep_" + key] = self.overall_sums[key]/ self.i
      except KeyError:
        pass
    recipe.curr_obs.update(res)
