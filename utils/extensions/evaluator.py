from extension import Extension

class Evaluator(Extension):

  def __call__(self, recipe):
    model = recipe.model
    source = recipe.source
    loss = model(next(source), train=False).data[0]
    recipe.curr_obs['val_ce'] = loss

