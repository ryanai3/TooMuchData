from os import path

from extensions import Extension

class SaveModel(Extension):

  def __init__(self, out_dir, fmt_str="snapshot"):
    self.fmt_str = fmt_str
    self.out_dir = out_dir

  def __call__(self, recipe):
    basename = self.fmt_str.format(recipe.i)
    fname = path.join(out_dir, basename)
    torch.save(recipe.model.state_dict(), fname)  

