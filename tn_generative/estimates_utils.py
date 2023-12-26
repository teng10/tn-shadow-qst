"""Utilities for estimating physical quantities from dataset or mps."""

from typing import Sequence

import numpy as np
import xarray as xr
import quimb.tensor as qtn

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
  if len(np.unique(mpo.bond_sizes())) != 1:
    raise ValueError(f'MPO of {mpo.bond_sizes()=} is not a product state, \
        can not slice.'
    )
  subsystem_indices = [i for i in range(mpo.L) if not np.allclose(
      mpo.arrays[i], np.array([[1., 0.], [0., 1.]])
  )]  # indices of non-identity subsystem.
  mpo_arrays_subsystem = [mpo.arrays[i] for i in subsystem_indices]
  # TODO: this is a hacky way to deal with the case where the subsystem is.
  if len(mpo_arrays_subsystem[0].shape) == 4:
    first_array = mpo_arrays_subsystem[0][0, ...]
    mpo_arrays_subsystem = [first_array, ] + mpo_arrays_subsystem[1:]
  if len(mpo_arrays_subsystem[-1].shape) == 4:
    last_array = mpo_arrays_subsystem[-1][:, 0, ...]
    mpo_arrays_subsystem = mpo_arrays_subsystem[:-1] + [last_array, ]
  if return_indices:
    return qtn.MatrixProductOperator(mpo_arrays_subsystem), subsystem_indices
  else:
    return qtn.MatrixProductOperator(mpo_arrays_subsystem)
  

def _construct_reduced_density_matrix(
    ds: xr.Dataset,
    subsystem: Sequence[int],
) -> qtn.MatrixProductOperator:
  """Constructs reduced density matrix for `subsystem` from `ds`."""
  bitstrings = ds['measurement'].sel(site=subsystem).values
  basis = ds['basis'].sel(site=subsystem).values
  # TODO: reconstruct reduced density matrix from `bitstrings` and `basis`.
  # need something like \sum U|b><b|U^\dagger.
  # TODO: convert reduced density matrix to MPO.
  raise NotImplementedError('TODO: reconstruct reduced density matrix.') 


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
