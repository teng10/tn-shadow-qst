"""Utilities for estimating physical quantities from dataset or mps."""

from typing import Callable, Sequence
import functools

import numpy as np
import xarray as xr
import quimb.tensor as qtn
import quimb as qu

from tn_generative import mps_utils
from tn_generative import physical_systems

PhysicalSystem = physical_systems.PhysicalSystem


def _extract_non_identity_mpo(
    mpo: qtn.MatrixProductOperator,
    return_indices: bool=False,
):
  """Extract the non-identity subsystem of an MPO.
  
  Args:
    mpo: an MPO of full system.

  Returns:
    an MPO of the non-identity acting on a subsystem.
  """
  def add_identity_tags(mpo, tag='I_like'):
    """Returns mpo with operators that are exactly identity tagged."""
    [x.add_tag(tag) for x in mpo if np.allclose(x.data, qu.pauli('I'))]
    return mpo
  non_identity = add_identity_tags(mpo).select(
      'I_like', which='!all'
  ).squeeze() # Select non-identity tensors
  indices = [int(tag[1:]) for tag in non_identity.tags] # Extract indices
  if return_indices:
    return non_identity, indices
  else:
    return non_identity
  

def stream_avg(ds: xr.Dataset, fn: Callable, batch_size: int=50):
  """Compute the streaming average of a function fn over a dataset ds.
  """
  # Compute the number of batches
  num_batches = ds.sizes['sample'] // batch_size
  # Compute the streaming average
  first_batch = ds.isel(sample=slice(0, batch_size))
  avg = fn(first_batch)
  for i in range(1, num_batches):
    # Compute the batch average
    batch = ds.isel(sample=slice(i*batch_size, (i+1)*batch_size))
    avg += fn(batch)
  # Compute the average of the remainder
  remainder = ds.isel(sample=slice(num_batches*batch_size, None))
  avg += fn(remainder)
  avg /= num_batches + 1
  return avg


# def _construct_reduced_density_matrix(
#     ds: xr.Dataset,
#     subsystem: Sequence[int],
# ) -> qtn.MatrixProductOperator:
#   """Constructs reduced density matrix for `subsystem` from `ds`."""
#   bitstrings = ds['measurement'].sel(site=subsystem).values
#   bases = ds['basis'].sel(site=subsystem).values
#   # TODO: reconstruct reduced density matrix from `bitstrings` and `basis`.
#   # need something like \sum U|b><b|U^\dagger.
#   # TODO: add streaming option for large datasets.
#   state_single_shots = []
#   for bitstring, basis in zip(bitstrings, bases):
#     mps_from_bitstring = qtn.MPS_computational_state(bitstring)
#     bitstring_dense = mps_from_bitstring.to_dense()
#     bitstring_dm = np.outer(bitstring_dense, np.conj(bitstring_dense))
#     # convert basis to MPO dense.
#     basis_unitary = (mps_utils.z_to_basis_mpo(basis)).to_dense()
#     state_dm = basis_unitary @ bitstring_dm @ basis_unitary.conj().T
#     state_single_shots.append(state_dm)
#   return np.mean(state_single_shots, axis=0)

# TODO: currently raises error...
def _construct_reduced_density_matrix(
    full_ds: xr.Dataset, subsystem: Sequence[int],
) -> qtn.MatrixProductOperator:
  """Constructs reduced density matrix for `subsystem` from `full_ds`."""
  def _reconstruction_dm_batch(
      ds: xr.Dataset, subsystem: Sequence[int]
    ) -> np.ndarray:
    """Reconstruct the density matrix from batched single shot data."""
    bitstrings = ds['measurement'].sel(site=subsystem).values
    bases = ds['basis'].sel(site=subsystem).values
    state_single_shots = []
    for bitstring, basis in zip(bitstrings, bases):
      mps_from_bitstring = qtn.MPS_computational_state(bitstring)
      bitstring_dense = mps_from_bitstring.to_dense()
      bitstring_dm = np.outer(bitstring_dense, np.conj(bitstring_dense))
      # convert basis to MPO dense.
      basis_unitary = (mps_utils.z_to_basis_mpo(basis)).to_dense()
      state_dm = basis_unitary @ bitstring_dm @ basis_unitary.conj().T
      state_single_shots.append(state_dm)
    return np.mean(state_single_shots, axis=0)
  
  _reconstruction_dm_batch_fn = functools.partial(
      _reconstruction_dm_batch, subsystem=subsystem
  )
  return stream_avg(full_ds, _reconstruction_dm_batch_fn)


def estimate_from_dataset(
    ds: xr.Dataset,
    mpo: qtn.MatrixProductOperator,
):
  """Estimates physical quantities from `ds`.
  
  Args:
    ds: dataset containing `measurement`, `basis`.
    mpo: MPO to estimate expectation value.
  """
  # STEP 0: extract indices of subsystems from `mpos`.
  sub_mpo, sub_indices = _extract_non_identity_mpo(mpo, return_indices=True)
  # ISSUE: this is not trivial, since `mpo` is not necessarily a product state.
  # STEP 1: estimate reduced density matrix for each subsystem from `ds`.
  rdm_mpo = _construct_reduced_density_matrix(ds, sub_indices)
  # STEP 2: estimate expectation value of `mpo` from `ds`.
  return (sub_mpo.apply(rdm_mpo)).trace()


def estimate_observable(
  train_ds: xr.Dataset,
  mpo: qtn.MatrixProductOperator,
  method: str = 'mps',
  tolerance: float = 1e-6,
) -> float:
  """Estimates expectation value of `mpo` with `train_ds` using `method`.

  Args:
    train_ds: dataset from which to estimate expectation value.
    mpo: MPO to estimate expectation value.
    method: method to use for estimation. Should we either `mps`, `shadow` or
    `placeholder`. Default is exact computation using 'mps'.

  Return:
    estimated expectation value.
  """
  def is_approximately_real(number):
    return abs(number.imag) < tolerance
  if method == 'placeholder':
    return 1.
  elif method == 'mps':
    mps = mps_utils.xarray_to_mps(train_ds)
    expectation_val = (mps.H @ (mpo.apply(mps)))
    if not is_approximately_real(expectation_val):
      raise ValueError(f'{expectation_val=} is not real.')
    return expectation_val.real
  elif method == 'shadow':
    raise NotImplementedError(f'{method=} not implemented.')
  else:
    raise ValueError(f'Unexpected estimation method {method}.')


def estimate_density_matrix(
  train_ds: xr.Dataset,
  subsystem: Sequence[int],
  method: str = 'mps',
) -> qtn.MatrixProductOperator:
  """Estimates reduced density matrix as `mpo` from `train_ds` using `method`.

  Args:
    trains_ds: dataset from which to estimate reduce density matrix.
    subsystem: subsystem indices for which to estimate reduced density matrix.
    method: method to use for estimation. Should we either `mps`, `shadow` or
    `placeholder`. Default is exact computation using 'mps'.

  Return:
    Reduced density matrix MPO.
  """
  if method == 'mps':
    mps = mps_utils.xarray_to_mps(train_ds)
    return mps.partial_trace(subsystem, rescale_sites=True)
  elif method == 'shadow':
    raise NotImplementedError(f'{method=} not implemented.')
  else:
    raise ValueError(f'Unexpected estimation method {method}.')
