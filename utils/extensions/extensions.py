class Extension():
  def __init__(self):
    raise NotImplementedError

  def __call__(self, recipe):
    raise NotImplementedError

  def with_interval(self, interval):
    return IntervalTriggeredExtension(self, interval)

class IntervalTriggeredExtension(Extension):

  def __init__(self, extension, every_n):
    self.extension = extension
    self.every_n = every_n

  def __call__(self, recipe):
    if recipe.i % self.every_n == 0:
      self.extension(recipe)
