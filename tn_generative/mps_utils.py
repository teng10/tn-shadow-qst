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

def amplitude_via_contraction(
    mps: qtn.MatrixProductState,
    measurement: jax.Array,
    basis: jax.Array | None = None,
) -> float | complex:
  """Computes `mps` amplitude for `measurement` with local `basis` rotations."""
  if measurement.shape != (mps.L,):
    raise ValueError(f'Cannot contract {mps.L=} with {measurement.shape=}.')
  arrays = jax.nn.one_hot(measurement, mps.phys_dim())
  arrays = [jnp.squeeze(x) for x in jnp.split(arrays, mps.L)] # changed x[0]
  bit_state = qtn.MPS_product_state(arrays)  # one-hot `measurement` MPS.
  if basis is not None:
    rotation_mpo = z_to_basis_mpo(basis)
    bit_state = rotation_mpo.apply(bit_state)  # rotate to local bases.
  return (mps | bit_state) ^ ...
