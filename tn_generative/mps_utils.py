"""Helper functions for manipulating MPS objects."""
import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import quimb as qmb
import quimb.tensor as qtn

from tn_generative import types

Array = types.Array
EST_REGISTRY = {}  # define registry for estimator functions.

HADAMARD = qmb.gen.operators.hadamard()
Y_HADAMARD = np.array([[0.0, -1.0j], [1.0j, 0.0]])
EYE = qmb.gen.operators.eye(2)


def z_to_basis_mpo(basis: Array) -> qtn.MatrixProductOperator:
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
  # convention: (x, y, z): (0, 1, 2).
  all_ops = jnp.stack([hadamard_ops, y_hadamard_ops, eye_ops])
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
  arrays = [jnp.squeeze(x) for x in jnp.split(arrays, mps.L)]
  bit_state = qtn.MPS_product_state(arrays)  # one-hot `measurement` MPS.
  if basis is not None:
    rotation_mpo = z_to_basis_mpo(basis)
    bit_state = rotation_mpo.apply(bit_state)  # rotate to local bases.
  return (mps | bit_state) ^ ...


def uniform_normalize(mps: qtn.MatrixProductState) -> qtn.MatrixProductState:
  """Normalizes `mps` by uniformly adjusting parameters of all tensors."""
  mps_copy = mps.copy()
  nfact = (mps_copy.H @ mps_copy)**0.5
  return mps_copy.multiply(1 / nfact, spread_over='all')


def uniform_param_normalize(mps_arrays: Sequence[Array])-> Sequence[Array]:
  """Normalizes `mps_arrays` by calling `uniform_normalize` on mps."""
  mps = qtn.MatrixProductState(arrays=mps_arrays)
  return uniform_normalize(mps).arrays


def mps_to_xarray(mps: qtn.MatrixProductState) -> xr.Dataset:
  """Packages `mps` parameters to xarray.Dataset."""
  if mps.L < 3:
    raise ValueError(f'Cannot convert {mps.L=} to xarray.Dataset.')
  mps = mps.copy()  # avoid modifying outside copy.
  # TODO(yteng) consider extending this to support non-default reconstruction.
  mps.permute_arrays('lrp')  # left, right, physical indices.
  mps.expand_bond_dimension(mps.max_bond())  # for simplicity keep max dim.
  mps_arrays = mps.arrays
  bulk_array = np.stack(mps_arrays[1:-1])
  bulk_dims = ('bulk_site', 'left_ind', 'right_ind', 'phys_ind')
  return xr.Dataset(
      data_vars={
          'left_tensor': (('right_ind', 'phys_ind'), mps_arrays[0]),
          'bulk_tensor': (bulk_dims, bulk_array),
          'right_tensor': (('left_ind', 'phys_ind'), mps_arrays[-1]),
      },
      coords={
          'left_ind': np.arange(mps.max_bond()),
          'right_ind': np.arange(mps.max_bond()),
          'phys_ind': np.arange(mps.phys_dim()),
          'bulk_site': np.arange(1, mps.L - 1),
      }
  )


def xarray_to_mps(ds: xr.Dataset) -> qtn.MatrixProductState:
  """Converts MPS arrays packaged in `ds` to `qtn.MatrixProductState`."""
  ds = ds.transpose('bulk_site', 'left_ind', 'right_ind', 'phys_ind', ...)
  bulk_arrays = np.split(ds.bulk_tensor.values, ds.sizes['bulk_site'])
  bulk_arrays = [x[0, ...] for x in bulk_arrays]
  mps_arrays = [ds.left_tensor.values] + bulk_arrays + [ds.right_tensor.values]
  return qtn.MatrixProductState(mps_arrays)


def _register_est_fn(get_est_fn, est_name: str):
  """Registers `get_est_fn` in global `EST_REGISTRY`."""
  registered_fn = EST_REGISTRY.get(est_name, None)
  if registered_fn is None:
    EST_REGISTRY[est_name] = get_est_fn
  else:
    if registered_fn != get_est_fn:
      raise ValueError(f'{est_name} is already registerd {registered_fn}.')


register_est_fn = lambda name: functools.partial(
    _register_est_fn, est_name=name
)


@register_est_fn('dmrg')
def get_estimator_dmrg(ds, mpo):
  """An estimator function that uses MPS from DMRG to compute expectation value.
  """
  # COMMENT(YT): do you have to use the wrapper function here? Silly, but I think yes.
  def estimator_fn_dmrg(ds, mpo):
    mps = xarray_to_mps(ds)
    return (mps.H | (mpo.apply(mps))) ^ ...
  return estimator_fn_dmrg(ds, mpo)


@register_est_fn('placeholder')
def get_estimator_placeholder(ds, mpo):
  """A placeholder estimator function that always returns 1.0."""
  estimator_plcaeholder = lambda ds, mpo: 1.
  return estimator_plcaeholder(ds, mpo)


@register_est_fn('shadow')
def get_estimator_shadow(ds, mpo):
  """An shadow estimator function that uses `measurement` and `basis` from
  dataset to compute expectation value.
  """
  raise NotImplementedError('Shadow estimator is not implemented yet.')
