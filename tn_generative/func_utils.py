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


def get_register_decorator(global_registry):
  """Returns a decorator that registers a function in `global_registry`."""
  def _register_decorator(fn, name_in_registry: str):
    """Registers `fn` in `registry` under `name_in_registry`."""
    registered_fn = global_registry.get(name_in_registry, None)
    if registered_fn is None:
      global_registry[name_in_registry] = fn
    else:
      if registered_fn != fn:
        raise ValueError(
            f'{name_in_registry=} is already registerd in {global_registry=} \
            as {registered_fn}.'
        )
  return lambda name: functools.partial(
      _register_decorator, name_in_registry=name
  )
