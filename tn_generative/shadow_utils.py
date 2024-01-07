"""Utilities for classical shadows."""
from typing import Callable, Sequence
import functools
import itertools

import numpy as np
import xarray as xr

from tn_generative import mps_utils

HADAMARD = mps_utils.HADAMARD
Y_HADAMARD = mps_utils.Y_HADAMARD
EYE = mps_utils.EYE
UP = np.array([[1., 0.], [0., 0.]])
DOWN = np.array([[0., 0.], [0., 1.]])
ALLUS = np.array([HADAMARD, Y_HADAMARD, EYE])
ALLSTATES = np.array([UP, DOWN])


def shadow_real_pauli_single_shot_vectorized(bits: np.ndarray, ids: np.ndarray
) -> np.ndarray:
  """Computes the single shot shadow for real states.

  Args:
    bits: bitstring or measurement (array of bits).
    us: basis (array of ids corresponding unitaries).

  Returns:
    local_shadows: array of local shadows.
  """
  local_states = ALLSTATES[bits]
  b_unitaries = ALLUS[ids]
  # Using NumPy operations for vectorization
  # This will compute all local shadows in one go.
  local_shadows = 2. * np.einsum(
      'ijk,ikl,ilm->ijm',
      b_unitaries.conj().transpose(0, 2, 1), # b_unitaries^dagger
      local_states, b_unitaries
  ) - 1./2. * np.eye(2)
  return local_shadows


def shadow_pauli_single_shot_vectorized(bits: np.ndarray, ids: np.ndarray
) -> np.ndarray:
  """Computes the single shot shadow for arbitrary states.

  Args:
    bits: bitstring or measurement (array of bits).
    us: basis (array of ids corresponding unitaries).

  Returns:
    local_shadows: array of local shadows.
  """
  local_states = ALLSTATES[bits]
  b_unitaries = ALLUS[ids]
  # Using NumPy operations for vectorization
  # This will compute all local shadows in one go.
  local_shadows = 3. * np.einsum(
      'ijk,ikl,ilm->ijm',
      b_unitaries.conj().transpose(0, 2, 1), # b_unitaries^dagger
      local_states, b_unitaries
  ) - np.eye(2)
  return local_shadows


def _get_shadow_single_shot_fn(ds: xr.Dataset
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Returns the single shot shadow function for `ds` based on `basis`."""
  unique_rows = np.unique(ds['basis'].values, axis=0)
  unique_ids = np.unique(ds['basis'].values)
  if set(unique_ids) == set([0, 2]) and unique_rows.shape[0] > 2:
    return shadow_real_pauli_single_shot_vectorized
  elif set(unique_ids) == set([0, 1, 2]) and unique_rows.shape[0] > 3:
    return shadow_pauli_single_shot_vectorized
  else:
    raise NotImplementedError('Only random XZ measurements are supported.')


def construct_subsystem_shadows(
    full_ds: xr.Dataset,
    subsystem: Sequence[int],
    shadow_single_shot_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
  """Stream to compute shadow states for `subsystem` from `full_ds`.

  Args:
    full_ds: dataset containing `measurement`, `basis`.
    subsystem: indices of subsystem.
    shadow_single_shot_fn: function to compute single shot shadow given
        two arrays: measurement outcome `bitstring` and unitary ids `basis`.

  Returns:
    `subsystem` shadow state.
  """
  def get_precomputed_single_shadows(n_sites):
    """Precompute all possible single shot shadows."""
    shadows_dict = {}
    for bits in itertools.product(*[tuple(range(2))] * n_sites):
      for ids in itertools.product(*[tuple(range(3))] * n_sites):
        bitstring = np.array(bits)
        basis = np.array(ids)
        shadow = shadow_single_shot_fn(bitstring, basis)
        shadows_dict[bits + ids] = functools.reduce(np.kron, list(shadow))
    return shadows_dict

  def _construct_shadow_mean(ds: xr.Dataset, precomputed_shadows: dict,
  ) -> np.ndarray:
    """Compute average shadow from single shot data.

    Args:
      ds: dataset containing measurements `bitstrings` and `bases`.
      precomputed_shadows: dictionary of all possible single shot shadows.

    Returns:
      average shadow state.
    """
    bitstrings = ds['measurement'].values
    bases = ds['basis'].values
    combined = np.concatenate([bitstrings, bases], axis=-1)
    uniques, counts = np.unique(combined, axis=0, return_counts=True)
    rdm = sum(c * precomputed_shadows[tuple(u)] for c, u in zip(counts, uniques))
    return rdm / sum(counts)

  subsystem_ds = full_ds.sel(site=subsystem)
  precomputed_shadows = get_precomputed_single_shadows(
      len(subsystem)
  ) # Precompute all possible single shot shadows.
  return _construct_shadow_mean(subsystem_ds, precomputed_shadows)
  # Compute the streaming mean over `sample` dimension.
  # TODO(YT): remove streaming mean, a lot slower than finding uniques.
  # return streaming_mean_over_dim(subsystem_ds, _get_subsystem_shadow, 'sample')
