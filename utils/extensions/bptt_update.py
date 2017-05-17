from recipe.extensions import Extension

class BPTTUpdate(Extension):
  def __init__(self, bptt_len):
    self.bptt_len = bptt_len

  def init_recipe(self, recipe):
    recipe.bptt_i = 0

  def __call__(self, recipe):
    model = recipe.model
    optimizer = recipe.optimizer
    source = recipe.source
    loss = 0.0
    for i in range(self.bptt_len):
      try:
        batch = next(source)
      except StopIteration:
        recipe.keep_going = False
        break
      loss += model(batch)
    loss.backward()
    optimizer.step()
    model.stop_bptt()
    recipe.curr_obs['ce'] = loss.data[0]
    recipe.i += self.bptt_len
