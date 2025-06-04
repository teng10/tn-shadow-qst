"""Noise utilities for postprocessing datasets"""
from __future__ import annotations
import functools
import copy
from typing import Tuple

import jax
import jax.numpy as jnp
import xarray as xr


NOISE_REGISTRY = {}  # define registry for noise functions.


NOISE_REGISTRY['none'] = None  # default noise is None.


def _register_noise_fn(get_noise_fn, name: str):
  """Registers `get_reg_fn` in global `NOISE_REGISTRY`."""
  registered_fn = NOISE_REGISTRY.get(name, None)
  if registered_fn is None:
    NOISE_REGISTRY[name] = get_noise_fn
  else:
    if registered_fn != get_noise_fn:
      raise ValueError(f'{name} is already registerd {registered_fn}.')


register_reg_fn = lambda name: functools.partial(
    _register_noise_fn, name=name
)

@register_reg_fn('bitflip')
def add_bitflip_noise_to_ds(
    key: jax.random.PRNGKeyArray,
    ds: xr.Dataset,
    noise_probabilities: Tuple[float, float],
) -> xr.Dataset:
  """Adds noise to the dataset by flipping bits with given probabilities.
  With `noise_probabilities`, a sample is flipped for 0, 1 bits.
  
  Args:
    ds: The input dataset containing the data.
    noise_probabilities: probabilities for flipping [0, 1] bits in any basis.
  Returns:
    noisy_ds: A new dataset with noise added.
  """
  noisy_ds = copy.deepcopy(ds)
  # Need a key for 0->1 and 1->0 flips, so we split the key into two parts
  noise_keys = jax.random.split(key, 2)
  
  for i in range(2):
    mask = ds.measurement == i  # mask for the current bit (0 or 1)
    flip = mask * jax.random.choice(
        noise_keys[i],
        a=2,
        shape=mask.shape,
        p=jnp.array([1 - noise_probabilities[i], noise_probabilities[i]]),
    )
    noisy_ds['measurement'] = (noisy_ds.measurement + flip ) % 2

  return noisy_ds
