"""Utilities for estimating physical quantities from dataset or mps."""

from typing import Sequence
import functools

import numpy as np
import xarray as xr
import quimb.tensor as qtn
import quimb as qu

from tn_generative import mps_utils
from tn_generative import physical_systems
from tn_generative import shadow_utils

PhysicalSystem = physical_systems.PhysicalSystem
PAULIMAP = {'X': 0, 'Y': 1, 'Z': 2, 'I': 3} # Mapping from pauli string to index.


def _extract_non_identity_mpo(
    mpo: qtn.MatrixProductOperator,
    return_indices: bool=False,
):
  """Extract the non-identity subsystem of an MPO.
  # TODO(YT): move to MPS utils.

  Args:
    mpo: an MPO of full system.

  Returns:
    an MPO of the non-identity acting on a subsystem.
  """
  def add_identity_tags(mpo, tag='I_like'):
    """Returns mpo with operators that are exactly identity tagged."""
    mpo = mpo.copy() # Don't modify original mpo.
    # Add tag to all identity tensors.
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


def _extract_pauli_prod_from_ds(
    ds: xr.Dataset,
    pauli: np.ndarray,
    indices = Sequence[int],
) -> xr.Dataset:
  """Compute pauli product for `pauli` of subsystem index `indices`.

  Args:
    ds: dataset containing `measurement`, `basis`.
    pauli: pauli word to estimate expectation value.
    indices: indices of subsystems of the pauli word.

  Returns:
    Pauli product of measurements for all samples in `pauli` basis.
  """
  # Step 0: extract measurements from `ds` for `indices` and `pauli`.
  ds_subsystem = ds.sel(site=indices)
  ds_pauli = ds_subsystem.where(ds_subsystem.basis == pauli, drop=True)
  # Step 1: compute the product of measurements for each sample.
  # TODO(YT): consider explicitly setting `axis`=1.
  def _pauli_prod(array, axis):
    return np.prod(-2. * array + 1, axis=axis) # 0->1, 1->-1
  pauli_prod = xr.apply_ufunc(
      _pauli_prod, ds_pauli.measurement,
      input_core_dims=[['site']], kwargs={'axis': -1}
  )
  return pauli_prod.dropna(dim='sample')


def estimate_expval_pauli_from_measurements(
    ds: xr.Dataset,
    pauli: np.ndarray,
    indices: Sequence[int],
    estimator: str = 'empirical',
    return_errbar: bool = False,
) -> float:
  """Estimate expectations of a pauli word from `ds` using direct measurements.

  This function filters out measurements in `ds` in `pauli` basis of subsystem.
  Then it computes the product of measurements for each sample and estimates the
  expectation value of the product using `estimator`.
  Note this should be used for non-randomized dataset for an accurate estimate.

  Args:
    ds: dataset containing `measurement`, `basis`.
    pauli: pauli word to estimate expectation value.
    indices: indices of subsystems of the pauli word.
    estimator: method for estimating expectation value.
    return_errbar: whether to return the errorbar of the estimator.

  Returns:
    Expectation value of `mpo` by selecting correct measurements from `ds`.
  """
  pauli_prod = _extract_pauli_prod_from_ds(ds, pauli, indices)
  # compute the average of the product of measurements.
  if estimator == 'empirical':
    mean = pauli_prod.mean(dim='sample').values
    std = pauli_prod.std(dim='sample').values
  elif estimator == 'mom':
    raise NotImplementedError(f'Pauli {estimator=} not implemented.')
  else:
    raise ValueError(f'Unknown pauli {estimator=}.')
  if return_errbar:
    return mean, std / np.sqrt(pauli_prod.sizes['sample'])
  else:
    return mean


def estimate_expval_mpo_from_shadow(
    ds: xr.Dataset,
    mpo: qtn.MatrixProductOperator,
    estimator: str = 'empirical',
) -> float:
  """Estimates expectation values of mpo operator from `ds` via shadow.

  Note: the `shadow` is in geneal not a true reduced density matrix but used for
  estimating expectation values.

  Args:
    ds: dataset containing `measurement`, `basis`.
    mpo: MPO to estimate expectation value.
    estimator: method for estimating expectation value. `empirical` or `mom`.

  Returns:
    Expectation value of `mpo` estimated from classical shadow.
  """
  # STEP 0: extract indices of subsystems from `mpos`.
  sub_mpo, sub_indices = _extract_non_identity_mpo(mpo, return_indices=True)
  # STEP 1: estimate reduced density matrix for each subsystem from `ds`.
  shadow_single_shot_fn = shadow_utils._get_shadow_single_shot_fn(ds)
  estimate_shadow_fn = functools.partial(
      shadow_utils.construct_subsystem_shadows,
      shadow_single_shot_fn=shadow_single_shot_fn, estimator=estimator,
  )
  subsystem_shadow = estimate_shadow_fn(ds, sub_indices)
  # STEP 2: estimate expectation value of `mpo` from `subsystem_shadow`.
  return (sub_mpo.to_dense() @ subsystem_shadow).trace()


def _extract_pauli_indices_from_mpo(
    mpo: qtn.MatrixProductOperator,
) -> Sequence[int]:
  """Extract pauli indices from `mpo` of a pauli string.

  # TODO(YT): move to MPS utils.
  Args:
    mpo: MPO to extract pauli indices.

  Returns:
    indices of pauli operators in `mpo`. {0, 1, 2, 3} -> {X, Y, Z, I}
  """
  pauli_indices = []
  for x in mpo:
    pauli_index = [
        PAULIMAP[tag] for tag in PAULIMAP.keys() if np.allclose(
            x.data, qu.pauli(tag)
        )
    ]
    if len(pauli_index) == 1:
      pauli_indices.append(pauli_index[0])
    else:
      raise ValueError(f'MPO {x=} is not a Pauli operator with {pauli_index=}')
  return pauli_indices


def estimate_observable(
  train_ds: xr.Dataset,
  mpo: qtn.MatrixProductOperator,
  method: str = 'mps',
  tolerance: float = 1e-6,
  estimator: str = 'empirical',
) -> float:
  """Estimates expectation value of `mpo` with `train_ds` using `method`.

  # TODO(YT): consider adding another wrapper function to select `method`
  # automatically based on `basis` in `train_ds`.
  # e.g. if `basis` is fixed XZ, then use `measurement` method.

  Args:
    train_ds: dataset from which to estimate expectation value.
    mpo: MPO to estimate expectation value.
    method: method to use for estimation. Should we either `mps`, `shadow` or
    `measurement`. Default is exact computation using 'mps'.
    tolerance: tolerance for checking whether expectation value is real.
    estimator: method for estimating expectation value. Options are
        `empirical` (default) or `mom` (median of means).

  Return:
    estimated expectation value.
  """
  def is_approximately_real(number):
    return abs(number.imag) < tolerance
  if method =='measurement':
    # TODO (YT): add warning if `ds` is not fixed basis. Otherwise this will
    # give an estimate with large errorbars.
    # STEP 0: extract indices of subsystems from `mpos`.
    sub_mpo, sub_indices = _extract_non_identity_mpo(mpo, return_indices=True)
    # STEP 1: compute the pauli word for `sub_mpo`
    pauli = np.array(_extract_pauli_indices_from_mpo(sub_mpo))
    # STEP 2: estimate expectation value of `pauli` from `ds`.
    return estimate_expval_pauli_from_measurements(
        train_ds, pauli, sub_indices, estimator=estimator,
    )
  elif method == 'shadow':
    # TODO (YT): add warning if `ds` is not random basis.
    return estimate_expval_mpo_from_shadow(train_ds, mpo, estimator)
  elif method == 'mps':
    mps = mps_utils.xarray_to_mps(train_ds)
    expectation_val = (mps.H @ (mpo.apply(mps)))
    if not is_approximately_real(expectation_val):
      raise ValueError(f'{expectation_val=} is not real.')
    return expectation_val.real
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
