"""Utilities for estimating physical quantities from dataset or mps."""

from typing import Sequence

import numpy as np
import xarray as xr
import quimb.tensor as qtn

from tn_generative import mps_utils
from tn_generative import physical_systems

PhysicalSystem = physical_systems.PhysicalSystem


def estimate_from_dataset(
    ds: xr.Dataset,
    mpo: qtn.MatrixProductOperator,
):
  """Estimates physical quantities from `ds`.
  
  Args:
    ds: dataset containing `measurement`, `basis`.
    mpo: MPO to estimate expectation value.
  """
  pass
  # STEP 0: extract indices of subsystems from `mpos`.
  # ISSUE: this is not trivial, since `mpo` is not necessarily a product state.
  # STEP 1: estimate reduced density matrix for each subsystem from `ds`.
  # STEP 2: estimate expectation value of `mpo` from `ds`.


def estimate_observable(
  train_ds: xr.Dataset,
  system: PhysicalSystem,
  method: str = 'mps',
  tolerance: float = 1e-6,
) -> float:
  """Estimates expectation value of `mpo` with `train_ds` using `method`.

  Args:
    train_ds: dataset from which to estimate expectation value.
    system: physical system to compute regularization terms for.
    method: method to use for estimation. Should we either `mps`, `shadow` or
    `placeholder`. Default is exact computation using 'mps'.

  Return:
    estimated expectation value.
  """
  def is_approximately_real(number):
    return abs(number.imag).all() < tolerance
  if method == 'placeholder':
    return 1.
  elif method == 'mps':
    ham_mpos = system.get_ham_mpos()
    mps = mps_utils.xarray_to_mps(train_ds)
    stabilizer_estimates = np.array([
          (mps.H @ (ham_mpo.apply(mps))) for ham_mpo in ham_mpos
      ])    
    if not is_approximately_real(stabilizer_estimates):
      raise ValueError(f'{stabilizer_estimates=} is not real.')
    return stabilizer_estimates.real
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
