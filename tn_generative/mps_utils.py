import jax
import jax.numpy as jnp
import numpy as np

# Quimb imports for tensor network manipulations
import quimb as qmb
import quimb.tensor as qtn
from tn_generative import typing


HADAMARD = qmb.gen.operators.hadamard()
Y_HADAMARD = np.array([[0.0, -1.0j], [1.0j, 0.0]])
EYE = qmb.gen.operators.eye(2)


def z_to_basis_mpo(basis: typing.Array) -> qtn.MatrixProductOperator:
  """Returns MPO that rotates from `z` to `basis` basis.

  Args:
    basis: 1d integer array specifying local bases with notation (0, 1, 2)
      corresponding to (X, Y, Z) bases.

  Returns: MPO that rotates from `z` to `basis` basis.

  """
  if basis.ndim != 1:
    raise ValueError(f'`basis` must be 1d, got {basis.shape=}')
  n_sites = basis.shape[0]
  hadamard_ops = jnp.stack([HADAMARD] * n_sites)
  y_hadamard_ops = jnp.stack([Y_HADAMARD] * n_sites)
  eye_ops = jnp.stack([EYE] * n_sites)
  all_ops = jnp.stack([hadamard_ops, y_hadamard_ops, eye_ops]) # (x, y, z): (0, 1, 2)
  one_hot_basis = jax.nn.one_hot(basis, 3, axis=0)
  operators = (jnp.expand_dims(one_hot_basis, (-2, -1)) * all_ops).sum(axis=0)
  per_site_ops = [jnp.squeeze(x) for x in jnp.split(operators, n_sites)]
  return qtn.MPO_product_operator(per_site_ops)