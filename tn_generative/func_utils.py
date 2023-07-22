#@title || Functional utilities (func_utils.py)
import functools

import numpy as np


def vectorized_method(otypes=None, signature=None):
  """Numpy vectorization wrapper that works with instance methods."""
  def decorator(fn):
    vectorized = np.vectorize(fn, otypes=otypes, signature=signature)
    @functools.wraps(fn)
    def wrapper(*args):
      return vectorized(*args)
    return wrapper
  return decorator
