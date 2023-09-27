"""Regularizers for training MPS models."""
import functools
from typing import Callable, Optional, Sequence, Union

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

from tn_generative import mps_utils
from tn_generative import physical_systems

PhysicalSystem = physical_systems.PhysicalSystem

REGULARIZER_REGISTRY = {}  # define registry for regularization functions.


def _register_reg_fn(get_reg_fn, name: str):
  """Registers `get_reg_fn` in global `REGULARIZER_REGISTRY`."""
  registered_fn = REGULARIZER_REGISTRY.get(name, None)
  if registered_fn is None:
    REGULARIZER_REGISTRY[name] = get_reg_fn
  else:
    if registered_fn != get_reg_fn:
      raise ValueError(f'{name} is already registerd {registered_fn}.')


register_reg_fn = lambda name: functools.partial(
    _register_reg_fn, name=name
)


@register_reg_fn('hamiltonian')
def get_hamiltonian_reg_fn(
    system: PhysicalSystem,
    train_ds: xr.Dataset,
    estimator: str = 'mps',
    beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for terms in a hamiltonian.

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    estimator: method used to compute expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  ham_mpos = system.get_ham_mpos()
  estimator_fn = functools.partial(
        mps_utils.estimate_observable, method=estimator
    )
  stabilizer_estimates = np.array([
      estimator_fn(train_ds, ham_mpo) for ham_mpo in ham_mpos
  ])
  def reg_fn(mps_arrays: Sequence[jax.Array]):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    stabilizer_expectations = jnp.array([
        (mps.H @ (s.apply(mps))) for s in ham_mpos
    ])
    return jnp.sum(
        beta * jnp.abs(stabilizer_expectations - stabilizer_estimates)**2
    )
  return reg_fn


@register_reg_fn('pauli_z')
def get_pauli_z_reg_fn(
  system: PhysicalSystem,
  ds: xr.Dataset,
  estimator: str = 'mps',
  beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for pauli z operators.

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    estimator: method used to compute expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  pauli_z_mpos = system.get_obs_mpos(
      [(1., ('z', i)) for i in range(system.n_sites)]
  )
  estimator_fn = functools.partial(
        mps_utils.estimate_observable, method=estimator
    )  
  pauli_z_estimates = np.array(
      [estimator_fn(ds, pauli_z) for pauli_z in pauli_z_mpos]
  )
  def reg_fn(mps_arrays):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    pauli_z_expectations = jnp.array([
        (mps.H @ (sz.apply(mps))) for sz in pauli_z_mpos
    ])
    return jnp.sum(
        beta * jnp.abs(pauli_z_expectations - pauli_z_estimates)**2
    )
  return reg_fn


@register_reg_fn('density_matrix')
def get_hamiltonian_reg_fn(
    system: PhysicalSystem,
    train_ds: xr.Dataset,
    estimator: str = 'mps',
    beta: Optional[Union[np.ndarray, float]] = 1.,
    subsystem_kwargs: Optional[dict] = {},
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for terms in a hamiltonian.

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    estimator: method used to compute expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.
    subsystem_kwargs: kwargs for `system.get_subsystems`.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  subsystems = system.get_subsystems(**subsystem_kwargs)  #TODO(YT): implement this in physical systems.
  mps_train = mps_utils.xarray_to_mps(train_ds)
  estimator_fn = functools.partial(
        mps_utils.estimate_density_matrices, method=estimator
    )
  reduced_density_matrices_estimates = [
      estimator_fn(mps_train, subsystem) for subsystem in subsystems
  ]
  def reg_fn(mps_arrays: Sequence[jax.Array]):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    reduced_density_matrices = [
        mps.partial_trace(subsystem, rescale_sites=False)
        for subsystem in subsystems
    ]
    return jnp.mean(
        beta * (
            [(rho1 - rho2).trace() for rho1, rho2 in zip(
                reduced_density_matrices, reduced_density_matrices_estimates)
            ]
        )
    )
  return reg_fn
