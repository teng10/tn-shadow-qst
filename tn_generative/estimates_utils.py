"""Utilities for estimating physical quantities from dataset or mps."""

from typing import Sequence
import functools

import numpy as np
import xarray as xr
import quimb.tensor as qtn
import quimb as qu
from pennylane.pauli import pauli_decompose, pauli_word_to_string

from tn_generative import mps_utils
from tn_generative import physical_systems
from tn_generative import shadow_utils

PhysicalSystem = physical_systems.PhysicalSystem
PAULIMAP = {'X': 0, 'Y': 1, 'Z': 2} # Mapping from pauli string to index.


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


def estimate_expval_pauli_from_measurements(
    ds: xr.Dataset,
    pauli: np.ndarray,
    indices: Sequence[int],
    estimator: str = 'empirical',
    return_err: bool = False,
) -> float:
  """Estimate expectations of a pauli word from `ds` using direct measurements.

  Note: this function direclty computes the average of measurement outcomes.
  It should be used when the number of measurements in pauli is large enough.
  `estimator` can be one of: `empirical`, `median of means`, etc.
  Errorbar is standard deviation / sqrt(number of samples).
  
  Args:
    ds: dataset containing `measurement`, `basis`.
    pauli: pauli word to estimate expectation value.
    indices: indices of subsystems of the pauli word.
    estimator: method for estimating expectation value.
    return_err: whether to return the errorbar of the estimator.
  
  Returns:
    Expectation value of `mpo` by selecting correct measurements from `ds`.
  """
  # Step 0: extract measurements from `ds` for `indices` and `pauli`.
  ds_subsystem = ds.sel(site=indices)
  ds_pauli = ds_subsystem.where(ds_subsystem.basis == pauli, drop=True)
  # Step 1: compute the product of measurements for each sample.
  def _pauli_prod(array, axis):
    return np.prod(-2. * array + 1, axis=axis) # 0->1, 1->-1
  pauli_prod = xr.apply_ufunc(
      _pauli_prod, ds_pauli.measurement,
      input_core_dims=[['site']], kwargs={'axis': -1}
  )
  pauli_prod = pauli_prod.dropna(dim='sample')
  # Step 2: compute the average of the product of measurements.
  if estimator == 'empirical':
    mean = pauli_prod.mean(dim='sample').values
    std = pauli_prod.std(dim='sample').values
  elif estimator == 'median of means':
    raise NotImplementedError(f'Pauli {estimator=} not implemented.')
  else:
    raise ValueError(f'Unknown pauli {estimator=}.')
  if return_err:
    return mean, std / np.sqrt(pauli_prod.shape[0])
  else:
    return mean


def estimate_expval_mpo_from_dataset(
    ds: xr.Dataset,
    mpo: qtn.MatrixProductOperator,
    method: str,
) -> float:
  """Estimates expectation values of physical quantities from `ds`.
  Note: the `shadow` is in geneal not a true reduced density matrix but used for
  estimating expectation values.

  Args:
    ds: dataset containing `measurement`, `basis`.
    mpo: MPO to estimate expectation value.
    method: method (`shadow` or `measurement`) to estimate expectation value.

  Returns:
    Expectation value of `mpo` estimated from `ds`.
  """
  # STEP 0: extract indices of subsystems from `mpos`.
  sub_mpo, sub_indices = _extract_non_identity_mpo(mpo, return_indices=True)
  if method == 'shadow':
    # STEP 1: estimate reduced density matrix for each subsystem from `ds`.
    shadow_single_shot_fn = shadow_utils._get_shadow_single_shot_fn(ds)
    estimate_shadow_fn = functools.partial(
        shadow_utils.construct_subsystem_shadows,
        shadow_single_shot_fn=shadow_single_shot_fn
    )
    subsystem_shadow = estimate_shadow_fn(ds, sub_indices)
    # STEP 2: estimate expectation value of `mpo` from `subsystem_shadow`.
    return (sub_mpo.to_dense() @ subsystem_shadow).trace()
  elif method == 'measurement':
    # STEP 1: compute the pauli word for `sub_mpo`
    paulis_from_mpo = pauli_decompose(sub_mpo.to_dense())
    pauli_string = pauli_word_to_string(paulis_from_mpo)
    pauli = np.array([PAULIMAP[pauli] for pauli in pauli_string])
    # STEP 2: estimate expectation value of `pauli` from `ds`.
    return estimate_expval_pauli_from_measurements(
        ds, pauli, sub_indices, estimator='empirical'
    )
  else:
    raise NotImplementedError(f'Estimation method {method} not implemented.')


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
  if method == 'measurement':
    return 1.
  elif method == 'mps':
    mps = mps_utils.xarray_to_mps(train_ds)
    expectation_val = (mps.H @ (mpo.apply(mps)))
    if not is_approximately_real(expectation_val):
      raise ValueError(f'{expectation_val=} is not real.')
    return expectation_val.real
  elif method == 'shadow':
    return estimate_expval_mpo_from_dataset(train_ds, mpo, method)
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
