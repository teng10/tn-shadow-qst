"""Data generation for training MPS models."""
import functools
from typing import Callable, Optional, Sequence, Union

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

from tn_generative import physical_systems

PhysicalSystem = physical_systems.PhysicalSystem

TASK_REGISTRY = {}  # define registry for task hamiltonians.
REG_REGISTRY = {}  # define registry for regularization functions.


def _register_task(get_task_system_fn, task_name: str):
  """Registers `get_task_system_fn` in global `TASK_REGISTRY`."""
  registered_fn = TASK_REGISTRY.get(task_name, None)
  if registered_fn is None:
    TASK_REGISTRY[task_name] = get_task_system_fn
  else:
    if registered_fn != get_task_system_fn:
      raise ValueError(f'{task_name} is already registerd {registered_fn}.')


register_task = lambda name: functools.partial(_register_task, task_name=name)


@register_task('surface_code')
def get_surface_code(
    size_x: int,
    size_y: int,
    coupling_value: float = 1.0,
    onsite_z_field: float = 0.,
) -> PhysicalSystem:
  """Generates surface code for `[size_x, size_y]` domain with specification
  of stabilizer coupling `coupling_value` and external field `onsite_z_field`.
  """
  return physical_systems.SurfaceCode(
      size_x, size_y, coupling_value, onsite_z_field
  )


def _register_reg_fn(get_reg_fn, reg_name: str):
  """Registers `get_reg_fn` in global `REG_REGISTRY`."""
  registered_fn = REG_REGISTRY.get(reg_name, None)
  if registered_fn is None:
    REG_REGISTRY[reg_name] = get_reg_fn
  else:
    if registered_fn != get_reg_fn:
      raise ValueError(f'{reg_name} is already registerd {registered_fn}.')


register_reg_fn = lambda name: functools.partial(
    _register_reg_fn, reg_name=name
)


@register_reg_fn('surface_code')
def get_surface_code_reg_fn(
    system: PhysicalSystem,
    train_ds: xr.Dataset,
    estimator_fn: Callable[[xr.Dataset, qtn.MatrixProductOperator], np.ndarray],
    beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for surface code stabilizers.

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    estimator_fn: function that computes expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  stabilizer_mpos = system.get_obs_mpos()
  stabilizer_estimates = np.array([
      estimator_fn(train_ds, stabilizer) for stabilizer in stabilizer_mpos
  ])
  def reg_fn(mps_arrays: Sequence[jax.Array]):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    stabilizer_expectations = jnp.array([
        (mps.H @ (s.apply(mps))) for s in stabilizer_mpos
    ])
    return jnp.sum(
        beta * jnp.abs(stabilizer_expectations - stabilizer_estimates)**2
    )
  return reg_fn


@register_reg_fn('pauli_z')
def get_pauli_z_reg_fn(
  system: PhysicalSystem,
  ds: xr.Dataset,
  estimator_fn: Callable[[xr.Dataset, qtn.MatrixProductOperator], np.ndarray],
  beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for pauli z operators.

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    estimator_fn: function that computes expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  pauli_z_mpos = system.get_obs_mpos(
      [(1., ('z', i)) for i in range(system.n_sites)]
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
