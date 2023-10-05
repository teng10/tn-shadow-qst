"""Functional utilities."""
import functools

import numpy as np
import jax


shape_structure = lambda tree: jax.tree_util.tree_map(lambda x: x.shape, tree)


def vectorized_method(otypes=None, signature=None):
  """Numpy vectorization wrapper that works with instance methods."""
  def decorator(fn):
    vectorized = np.vectorize(fn, otypes=otypes, signature=signature)
    @functools.wraps(fn)
    def wrapper(*args):
      return vectorized(*args)
    return wrapper
  return decorator


def get_register_fn(registry):
  def _register_fn(get_fn, name: str):
    """Registers `get_fn` in `registry`."""
    registered_fn = registry.get(name, None)
    if registered_fn is None:
      registry[name] = get_fn
    else:
      if registered_fn != get_fn:
        raise ValueError(f'{name} is already registerd {registered_fn}.')
  register_fn = lambda name: functools.partial(
      _register_fn, name=name
  )
  return register_fn
