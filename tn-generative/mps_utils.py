import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
# Quimb imports for tensor network manipulations
import quimb as qmb
import quimb.tensor as qtn
from quimb.experimental import operatorbuilder as quimb_exp_op

Array = Union[np.ndarray, jnp.ndarray]
shape_structure = lambda tree: jax.tree_util.tree_map(lambda x: x.shape, tree)
#@title || MPO for rotating states in different bas (mpo_utils.py)

HADAMARD = qmb.gen.operators.hadamard()
Y_HADAMARD = np.roll(HADAMARD, 1, 0) * np.array([[1.0, 1.0j]])
EYE = qmb.gen.operators.eye(2)

def z_to_basis_mpo(basis: Array) -> qtn.MatrixProductOperator:
  """Returns MPO that rotates from `z` to `basis` basis.

  Args:
    basis: 1d integer array specifying local bases with notation (0, 1, 2)
      corresponding to (X, Y, Z) bases.

  Returns:

  """
  if basis.ndim != 1:
    raise ValueError(f'`basis` must be 1d, got {basis.shape=}')
  n_sites = basis.shape[0]
  hadamard_ops = jnp.stack([HADAMARD] * n_sites)
  y_hadamard_ops = jnp.stack([Y_HADAMARD] * n_sites)
  eye_ops = jnp.stack([EYE] * n_sites)
  all_ops = jnp.stack([hadamard_ops, y_hadamard_ops, eye_ops]) # (x, y, z): (0, 1, 2)
  # one_hot_basis = jax.nn.one_hot(basis % 3, 3, axis=0) #YT: added %3 to be consistent with the convention (x, y z)=(1, 2, 3)
  one_hot_basis = jax.nn.one_hot(basis, 3, axis=0)
  operators = (jnp.expand_dims(one_hot_basis, (-2, -1)) * all_ops).sum(axis=0)
  per_site_ops = [jnp.squeeze(x) for x in jnp.split(operators, n_sites)]
  return qtn.MPO_product_operator(per_site_ops)