from copy import deepcopy

class OrderedRecipe():
  def __init__(self, source, base_obs = {}, params = {}, orig_state = {}, extensions = []):
    self.i = 0 #  These can be overwritten by orig_state
    self.params = params # 
    self.base_obs = base_obs # 
    self.keep_going = True
    self.__dict__.update(orig_state)
    self.add_extension(extensions) # playing nice with orig_state if it contains extensions
  
  def run_ext_init(self, extension):
    try:
      extension.init_recipe(self)
    except AttributeError:
      pass

  def run_ext_inits(self):
    for ext in self.extensions:
      self.run_ext_init(extension)

  def add_extension(self, extension):
    self.extensions.append(extension)
    self.run_ext_init(extension)

  def add_extensions(self, extensions):
    for extension in extensions:
      self.add_extension(extension)

  def run(self):
    while self.keep_going:
      self.curr_obs = deepcopy(self.base_obs)
      for extension in extensions:
        extension(self)
