"""Helper functions for manipulating MPS objects."""
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import quimb as qmb
import quimb.tensor as qtn

from tn_generative import types

Array = types.Array

HADAMARD = qmb.gen.operators.hadamard()
Y_HADAMARD = 1./ np.sqrt(2.) * np.array([[-1.j, -1.], [1, 1.j]])
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


def _mps_to_expanded_tensors(mps: qtn.MatrixProductState) -> tuple[np.ndarray]:
  """
  Extract and expand tensors at the first and last sites of the MPS to 3-legs.

  Args:
    mps: MPS state.

  Returns:
    MPS arrays after expanding the first and last sites to 3-legs with
    indices (left virtual, right virtual, physical).
  """
  mps_tensors = mps.arrays
  # first tensor has indices (right virtual, physical)
  first_tensor = mps_tensors[0][np.newaxis, ...] # (1, right virtual, physical)
  # last tensor has indices (left virtual, physical)
  last_tensor = mps_tensors[-1][:, np.newaxis, :] # (left virtual, 1, physical)
  return (first_tensor, ) + mps_tensors[1:-1] + (last_tensor, )

def _mps_from_extended_tensors(
    mps_tensors: tuple[np.ndarray]
) -> qtn.MatrixProductState:
  """Convert the expanded tensors to MPS.

  Args:
    mps_tensors: MPS tensors after expanding the first and last sites to 3-legs.

  Returns:
    MPS state.
  """
  # Squeeze the first and last tensors
  # first tensor has indices (1, right virtual, physical)
  first_tensor_squeezed = mps_tensors[0][0, ...] # (right virtual, physical)
  # last tensor has indices (left virtual, 1, physical)
  last_tensor_squeezed = mps_tensors[-1][:, 0, :] # (left virtual, physical)
  return qtn.MatrixProductState(
      (first_tensor_squeezed,) + mps_tensors[1:-1] + (last_tensor_squeezed,)
  )


def transfer_matrices_adag_a(mps: qtn.MatrixProductState) -> list[np.ndarray]:
  """Compute the transfer matrices partially contracted of the MPS.

  Args:
    mps: MPS state.

  Returns:
    Transfer matrices of the MPS.
  """
  mps_tensors = _mps_to_expanded_tensors(mps)
  # Compute the transfer matrices
  return [np.einsum('ijk,lrk->jilr', A.conj(), A) for A in mps_tensors]

def contracted_transfer_matrices_left(
    mps: qtn.MatrixProductState
) -> list[np.ndarray]:
  """Compute the contracted transfer matrices from the left of the MPS.

  Args:
    mps: MPS state.

  Returns:
    Contracted transfer matrices of the MPS.
  """
  # sum over left bond of transfer matrices
  return [
    np.einsum('jklr, kl->jr', x, np.eye(x.shape[1]))
    for x in transfer_matrices_adag_a(mps)
  ]


def fix_gauge_arrays(mps: qtn.MatrixProductState):
  """Gauge fixing after canonicalization of the MPS.

  Args:
    mps: MPS state.

  Returns:
    MPS arrays after fixing the gauge.
  """
  def _unitary_gauge_fix(mps_tensors):
    """Fix the unitary gauge of A matrices at all sites."""
    unitaries = []
    tesnors_transformed = []
    tensors_transformed_rhs = []
    # Step 1: compute the unitaries that fix the gauge from RHS
    for a in mps_tensors:
      # Compute A^\dagger A contracted from left
      adag_a_left_contracted = np.einsum('jik,jlk->il', a.conj(), a)
      # Compute the eigenvectors of sum_i A_i^\dagger A_i
      # these are the unitaries that fix the gauge from RHS
      _, eigvecs = np.linalg.eigh(adag_a_left_contracted)
      unitaries.append(eigvecs)
      # Fix the gauge of A matrices
      # transform A matrices from RHS
      a_gauge_fixed_right = np.einsum('ijk,jl->ilk', a, eigvecs)
      tensors_transformed_rhs.append(a_gauge_fixed_right)
    # COMMENT: the following steps are not needed if we only want to compute the
    # schatten norm of contracted A^\dagger A matrices using the function
    # `compute_schatten_norm_Adag_A`,
    #  because A^\dagger A is invariant from the left unitaries.
    # Step 2: Fix the gauge of A matrices from LHS
    # the leftmost A matrix do not get transformed from LHS
    tesnors_transformed.append(tensors_transformed_rhs[0])
    for unitary, a in zip(unitaries[:-1], tensors_transformed_rhs[1:]):
      a_gauge_fixed_left = np.einsum('ji,jlk->ilk', unitary.conj(), a)
      tesnors_transformed.append(a_gauge_fixed_left)
    return tuple(tesnors_transformed)

  mps = mps.copy()
  mps = mps.canonize(0, cur_orthog=None) # right canonical form
  # Make all tensors 3-legs
  mps_tensors = _mps_to_expanded_tensors(mps)
  # for each physical sites, compute A matrix after fixing the gauge
  return _unitary_gauge_fix(mps_tensors)


def compute_schatten_norm_adag_a(
    mps_target: qtn.MatrixProductState,
    mps_result: qtn.MatrixProductState,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
  """Compute the Schatten norm of the difference between the \sum_sigma A+ A
  matrices of the target and result MPS.

  Args:
    mps_target: Target MPS.
    mps_result: Result MPS.
    distance_fn: Distance function.

  Returns:
    Schatten norm of the difference between the \sum_sigma A+ A matrices of the
    target and result MPS.
  """
  contracted_transfer_target = contracted_transfer_matrices_left(mps_target)
  contracted_transfer_result = contracted_transfer_matrices_left(mps_result)

  contracted_transfer_eigvals_target = [
      np.sort(np.linalg.eigvals(x)) for x in contracted_transfer_target
  ]
  contracted_transfer_eigvals_result = [
      np.sort(np.linalg.eigvals(x)) for x in contracted_transfer_result
  ]
  total_dim = sum([len(x) for x in contracted_transfer_eigvals_target])
  error_canonical_sites = jax.tree_map(
      distance_fn, contracted_transfer_eigvals_target,
      contracted_transfer_eigvals_result
  )
  return sum(error_canonical_sites) / total_dim
