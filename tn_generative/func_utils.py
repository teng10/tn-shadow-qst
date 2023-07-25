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
