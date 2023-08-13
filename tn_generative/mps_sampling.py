"""Sampling methods for MPS states."""
from typing import Tuple
import functools

import jax
import jax.numpy as jnp
import numpy as np

import quimb.tensor as qtn

from tn_generative import types
from tn_generative import mps_utils

Array = types.Array
SamplerFn = types.SamplerFn
MeasurementAndBasis = types.MeasurementAndBasis

SAMPLER_REGISTRY = {}  # Global registry for samplers.


def _register_sampler(sampler_fn: SamplerFn, sampler_name: str):
  """Registers `sampler_fn` in global `SAMPLER_REGISTRY`."""
  registered_fn = SAMPLER_REGISTRY.get(sampler_name, None)
  if registered_fn is None:
    SAMPLER_REGISTRY[sampler_name] = sampler_fn
  else:  # TODO(YT): check if this is the right way to do this, or overwrite?
    if registered_fn != sampler_fn:
      raise ValueError(f'{sampler_name} is already registerd {registered_fn}.')


register_sampler = lambda name: functools.partial(
    _register_sampler, sampler_name=name) # decorator for registering samplers.


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
    x_y_z_probabilities: Tuple[float, float, float],
    base_sample_fn: SamplerFn = gibbs_sampler,
) -> MeasurementAndBasis:
  """Draws a sample from `mps` in random X, Y or Z basis at each site.

  Samples `mps` in an X, Y or Z basis selected randomly at each site,
  with probabilities of `x_y_z_probabilities`.

  Args:
    key: random key used to draw a sample.
    mps: matrix product state in `z` basis from which to draw a sample.
    x_y_z_probabilities: probabilities for selecting X, Y or Z basis.
    base_sample_fn: sampler method. Default is gibbs_sampler.

  Returns:
    Tuple of mesurement sample and basis.
  """
  sample_key, basis_key = jax.random.split(key, 2)
  basis = jax.random.choice(basis_key, 3, [mps.L], p=x_y_z_probabilities)
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


def xz_neel_basis_sampler(
    key: jax.random.PRNGKeyArray,
    mps: qtn.MatrixProductState,
    neel_probabilities: Tuple[float, float],
    base_sample_fn: SamplerFn = gibbs_sampler,
) -> MeasurementAndBasis:
  """Draws a sample from `mps` in alternating X/Z basis selected randomly.

  Samples `mps` in an alternating X/Z basis which is selected randomly
  with probabilities neel_probabilities. 
  [XZX..., ZXZ...] are mapped to [0, 1].

  Args:
    key: random key used to draw a sample.
    mps: matrix product state in `z` basis from which to draw a sample.
    neel_probabilities: probabilities for selecting XZX.../ZXZ... basis.
    base_sample_fn: sampler method. Default is gibbs_sampler.

  Returns:
    Tuple of mesurement sample and basis.
  """
  neel_probabilities = jnp.asarray(neel_probabilities)
  sample_key, basis_key = jax.random.split(key, 2)
  basis_val_start = jax.random.choice(basis_key, np.arange(2), 
      p=neel_probabilities
  )
  basis = (jnp.arange(mps.L) + basis_val_start) % 2 * 2.
  mps = mps.copy()
  rotation_mpo = mps_utils.z_to_basis_mpo(basis)
  rotated_mps = rotation_mpo.apply(mps)
  return base_sample_fn(sample_key, rotated_mps), basis


register_sampler('x_basis_sampler')(
    functools.partial(fixed_basis_sampler, basis=0))
register_sampler('z_basis_sampler')(
    functools.partial(fixed_basis_sampler, basis=2))
register_sampler('xz_basis_sampler')(
    functools.partial(random_uniform_basis_sampler,
        x_y_z_probabilities=[0.5, 0.0, 0.5]
    ))
register_sampler('xz_neel_basis_sampler')(
    functools.partial(xz_neel_basis_sampler, neel_probabilities=[0.5, 0.5]))
register_sampler('x_or_z_basis_sampler')(
    functools.partial(random_basis_sampler,
        x_y_z_probabilities=[0.5, 0.0, 0.5]
    )
)
