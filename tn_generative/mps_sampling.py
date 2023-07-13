from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import quimb.tensor as qtn

from tn_generative import typing
from tn_generative import mps_utils

Array = typing.Array
SamplerFn = typing.SamplerFn
MeasurementAndBasis = typing.MeasurementAndBasis


def gibbs_sampler(
    key: jax.random.PRNGKey, 
    mps: qtn.MatrixProductState,
) -> Array:
  """Sample an observation from `mps` using gibbs sampling method.

  Args: 
    key: random key for sampling a measurement.
    mps: mps state from which a sample is drawn.

  Returns:
    outcomes of shape (L,) of integers in {0, 1} where L is the system size.
  """
  mps = mps.copy()  # let's not modify outside copy here.
  keys = jax.random.split(key, mps.L)
  mps.canonize(0, cur_orthog=None)  # start with right canonical form.
  outcomes = []
  site_idx = 0  # we will iterate over 0th site as we project out sites.
  for i, rng in enumerate(keys):
    L = mps.L  # current length of the MPS.
    t = mps[site_idx]  # current tensor for sampling.
    site_ind = mps.site_ind(site_idx)  # name of the site index.
    # diagonal of reduced density matrix corresponds to measurement probs.
    t_ii = t.contract(t.H, output_inds=(site_ind,))
    probs = jnp.real(t_ii.data)
    # sample 0 or 1 depending on the likelihood.
    outcome = jax.random.choice(rng, np.arange(mps.phys_dim()), [], p=probs)
    # project the outcome of the measurement.
    t.isel_({site_ind: outcome})
    # renormalize.
    t.modify(data=t.data / probs[outcome]**0.5)
    # contract projected tensor into the MPS and retag/reindex.
    if site_idx == L - 1:
      mps ^= slice(site_idx - 1, site_idx + 1)
    else:
      mps ^= slice(site_idx, site_idx + 2)
    for i in range(site_idx + 1, L):
      mps[i].reindex_({mps.site_ind(i): mps.site_ind(i - 1)})
      mps[i].retag_({mps.site_tag(i): mps.site_tag(i - 1)})
    mps._L = L - 1
    outcomes.append(outcome)
  return jnp.stack(outcomes)  


def fixed_basis_sampler(
    key: jax.random.PRNGKeyArray,
    mps: qtn.MatrixProductState,
    basis: Array | int,
    base_sample_fn: SamplerFn = gibbs_sampler,
) -> MeasurementAndBasis:
  """Draws a sample from `mps` in fixed basis specified by `basis`.

  Samples `mps` in a fixed `basis` by rotating `mps` to that basis and sampling
  from the resulting state. Basis is specified using [0, 1, 2] --> [X, Y, Z]
  mapping.

  Args:
    key: random key used to draw a sample.
    mps: matrix product state in `z` basis from which to draw a sample.
    basis: basis in which to sample specified as 1d array or int.
    base_sample_fn: sampler method. Default is gibbs_sampler.

  Returns:
    Tuple of mesurement sample and basis.
  """
  basis = jnp.asarray(basis)
  if basis.ndim > 1:
    raise ValueError(f'`basis` must be at most 1D, got: {basis.shape=}.')
  basis = jnp.broadcast_to(basis, [mps.L])
  mps = mps.copy()
  rotation_mpo = mps_utils.z_to_basis_mpo(basis)
  rotated_mps = rotation_mpo.apply(mps)
  return base_sample_fn(key, rotated_mps), basis


def random_basis_sampler(
    key: jax.random.PRNGKeyArray,
    mps: qtn.MatrixProductState,
    base_sample_fn: SamplerFn = gibbs_sampler,
) -> MeasurementAndBasis:
  """Draws a sample from `mps` in random X, Y or Z basis at each site.

  Samples `mps` in an X, Y or Z basis selected randomly at each site.

  Args:
    key: random key used to draw a sample.
    mps: matrix product state in `z` basis from which to draw a sample.
    base_sample_fn: sampler method. Default is gibbs_sampler.

  Returns:
    Tuple of mesurement sample and basis.
  """
  sample_key, basis_key = jax.random.split(key, 2)
  basis = jax.random.randint(basis_key, [mps.L], minval=0, maxval=3)
  mps = mps.copy()
  rotation_mpo = mps_utils.z_to_basis_mpo(basis)
  rotated_mps = rotation_mpo.apply(mps)
  return base_sample_fn(sample_key, rotated_mps), basis


def random_uniform_basis_sampler(
    key: jax.random.PRNGKeyArray,
    mps: qtn.MatrixProductState,
    x_y_z_probabilities: Tuple[float, float, float],
    base_sample_fn: SamplerFn = gibbs_sampler,
) -> MeasurementAndBasis:
  """Draws a sample from `mps` in uniformly X, Y or Z basis selected randomly.

  Samples `mps` in a uniform X, Y or Z basis which is selected randomly
  with probabilities x_y_z_probabilities. [X, Y, Z] are mapped to [0, 1, 2].

  Args:
    key: random key used to draw a sample.
    mps: matrix product state in `z` basis from which to draw a sample.
    x_y_z_probabilities: probabilities for selecting X, Y or Z basis.
    base_sample_fn: sampler method. Default is gibbs_sampler.

  Returns:
    Tuple of mesurement sample and basis.
  """
  x_y_z_probabilities = jnp.asarray(x_y_z_probabilities)
  sample_key, basis_key = jax.random.split(key, 2)
  basis_val = jax.random.choice(basis_key, jnp.arange(3), p=x_y_z_probabilities)
  basis = jnp.ones(mps.L).astype(int) * basis_val
  mps = mps.copy()
  rotation_mpo = mps_utils.z_to_basis_mpo(basis)
  rotated_mps = rotation_mpo.apply(mps)
  return base_sample_fn(sample_key, rotated_mps), basis
