"""Regularizers for training MPS models."""
import functools
from typing import Callable, Optional, Sequence, Union

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import quimb as qu
import quimb.tensor as qtn

from tn_generative import estimates_utils
from tn_generative import mps_utils
from tn_generative import physical_systems
from tn_generative import shadow_utils

PhysicalSystem = physical_systems.PhysicalSystem

REGULARIZER_REGISTRY = {}  # define registry for regularization functions.


REGULARIZER_REGISTRY['none'] = None  # default regularizer is None.


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
    method: str = 'mps',
    beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for non-onsite terms in a hamiltonian.

  Args:
    system: physical system where the dataset is generated from.
    train_ds: dataset containing `measurement`, `basis`.
    method: method used to compute expectation value of the regularization
        mpos and dataset `train_ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  def _get_multisite_mpos(
      mpos: Sequence[qtn.MatrixProductOperator],
  ): # Note: added 01/07/2024, could have led to different results.
    """Returns only multisite mpos from a list of mpos."""
    def _is_multisite(mpo: qtn.MatrixProductOperator):
      """Returns True if `mpo` is multisite."""
      subsystem_indices = [i for i in range(mpo.L) if not np.allclose(
            mpo.arrays[i], np.array([[1., 0.], [0., 1.]])
        )]
      return len(subsystem_indices) != 1

    def _is_identity(mpo: qtn.MatrixProductOperator):
      """Returns True if `mpo` is identity."""
      return all([np.allclose(x.data, qu.pauli('I')) for x in mpo])
        
    return [
        mpo for mpo in mpos if (_is_multisite(mpo) and not _is_identity(mpo))
    ]

  
  ham_mpos = system.get_ham_mpos()
  ham_multisite_mpos = _get_multisite_mpos(ham_mpos)
  method_fn = functools.partial(
        estimates_utils.estimate_observable, method=method
    )
  stabilizer_estimates = np.array([
      method_fn(train_ds, ham_mpo) for ham_mpo in ham_multisite_mpos
  ])
  def reg_fn(mps_arrays: Sequence[jax.Array]):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    stabilizer_expectations = jnp.array([
        (mps.H @ (s.apply(mps))) for s in ham_multisite_mpos
    ])
    return jnp.mean(
        beta * jnp.abs(stabilizer_expectations - stabilizer_estimates)**2
    )
  return reg_fn


@register_reg_fn('pauli_z')
def get_pauli_z_reg_fn(
  system: PhysicalSystem,
  train_ds: xr.Dataset,
  method: str = 'mps',
  beta: Optional[Union[np.ndarray, float]] = 1.,
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function for pauli z operators.

  Args:
    system: physical system where the dataset is generated from.
    train_ds: dataset containing `measurement`, `basis`.
    method: method used to compute expectation value of the regularization
        mpos and dataset `train_ds`.
    beta: regularization strength. Default is 1.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  pauli_z_mpos = system.get_obs_mpos(
      [(1., ('z', i)) for i in range(system.n_sites)]
  )
  method_fn = functools.partial(
        estimates_utils.estimate_observable, method=method
    )
  pauli_z_estimates = np.array(
      [method_fn(train_ds, pauli_z) for pauli_z in pauli_z_mpos]
  )
  def reg_fn(mps_arrays):
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    pauli_z_expectations = jnp.array([
        (mps.H @ (sz.apply(mps))) for sz in pauli_z_mpos
    ])
    return jnp.mean(
        beta * jnp.abs(pauli_z_expectations - pauli_z_estimates)**2
    )
  return reg_fn


def _get_subsystems(
    physical_system: PhysicalSystem,
    method: str,
    explicit_subsystems: Optional[list[Sequence[int]]] = None,
) -> list[Sequence[int]]:
  """Wrapper function to get subsystem indices.

  Args:
    physical_system: physical system where the dataset is generated from.
    method: method used to get subsystem indices.
    explicit_subsystems: subsystem indices returned if `method`=='explicit'.

  Returns:
    list of subsystem indices.
  """
  if method == 'default':
    try:
      subsystems = physical_system.get_subsystems()
    except NotImplementedError as x:
      raise x
  elif method == 'explicit':
    if explicit_subsystems is not None:
      subsystems = explicit_subsystems
      if np.max(subsystems) >= physical_system.n_sites:
        raise ValueError(f'{subsystems=} exceeds {physical_system.n_sites}.')
    else:
      raise ValueError(f'{method=} requires {explicit_subsystems=}.')
  else:
    raise ValueError(f'Unexpected {method=} is not implemented.')
  return subsystems


# TODO(YT): currently runs out of memory for ruby subsystem of size 6.
@register_reg_fn('reduced_density_matrices')
def get_density_reg_fn(
    system: PhysicalSystem,
    train_ds: xr.Dataset,
    method: str = 'mps',
    beta: Optional[Union[np.ndarray, float]] = 1.,
    subsystem_kwargs: Optional[dict] = {
        'method': 'default', 'explicit_subsystems': None
    },
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function using mpos of reduced density matrices.
  |\rho_target - \rho_model|_2

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    method: method used to compute expectation value of the regularization
        mpos and dataset `ds`.
    beta: regularization strength. Default is 1.
    subsystem_kwargs: kwargs for `system.get_subsystems`.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  subsystems = _get_subsystems(system, **subsystem_kwargs)
  method_fn = functools.partial(
      estimates_utils.estimate_density_matrix, method=method
  )
  reduced_density_matrices_estimates = [
      method_fn(train_ds, subsystem) for subsystem in subsystems
  ]
  def reg_fn(mps_arrays: Sequence[jax.Array]) -> float:
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    reduced_density_matrices = [
        mps.partial_trace(subsystem, rescale_sites=True)
        for subsystem in subsystems
    ]
    # Note: rescale_sites=True ensures physical tags are the same.
    # COMMENT(YT): Frobenius norm between the reduced density matrices.
    # Could also consider trace/nuclear distance.
    # Right now explicitly build reduced density matrices and use linear algebra
    # to compute Frobenius norm.
    # Could also use MPOs to compute the trace distance, but slower.
    return beta * jnp.mean(jnp.array(
        [jnp.linalg.norm((rho_1 - rho_2).to_dense(), ord='fro') for rho_1, rho_2
            in zip(reduced_density_matrices, reduced_density_matrices_estimates)
        ])
    )
  return reg_fn


@register_reg_fn('subsystem_xz_operators')
def get_subsystem_xz_pauli_reg_fn(
    system: PhysicalSystem,
    train_ds: xr.Dataset,
    beta: Optional[Union[np.ndarray, float]] = 1.,
    subsystem_kwargs: Optional[dict] = {
        'method': 'default', 'explicit_subsystems': None
    },
    paulis: Optional[str] = 'XZI',
    method: Optional[str] = 'shadow',
) -> Callable[[Sequence[jax.Array]], float]:
  """Returns regularization function using shadow of the subsystems.

  Using operator norm |\rho_{shadow} - \rho_{model, xz}|_2 (or schatten-2 norm).

  Args:
    system: physical system where the dataset is generated from.
    ds: dataset containing `measurement`, `basis`.
    beta: regularization strength. Default is 1.
    subsystem_kwargs: kwargs for `system.get_subsystems`.
        `method`: method used to get subsystem indices. 
            'default': use `system.get_subsystems`.
            'explicit': use `explicit_subsystems`.
    paulis: pauli operators used to construct the shadow.
    method: method for computing the projection. `shadow` or `mps`.

  Return:
    reg_fn: regularization function takes MPS arrays.
  """
  subsystems = _get_subsystems(system, **subsystem_kwargs)
  if method == 'shadow':
    # TODO (YT): consider add `estimator` for shadow method in
    # `shadow_utils.construct_subsystem_shadows`
    estimator_fn = functools.partial(
        shadow_utils.construct_subsystem_shadows, 
        shadow_single_shot_fn=shadow_utils._get_shadow_single_shot_fn(train_ds)
    )
    xz_estimates = [
        estimator_fn(train_ds, subsystem) for subsystem in subsystems
    ]
  elif method == 'mps':
    mps = mps_utils.xarray_to_mps(train_ds)
    xz_estimates = [
        mps_utils.construct_subsystem_operators(mps, subsystem, paulis) for 
        subsystem in subsystems
    ]
  else:
    raise ValueError(f'Unexpected {method=} is not implemented.')
  def reg_fn(mps_arrays: Sequence[jax.Array]) -> float:
    mps = qtn.MatrixProductState(arrays=mps_arrays)
    xz_estimates_model = [
        mps_utils.construct_subsystem_operators(mps, subsystem, paulis)
        for subsystem in subsystems
    ]
    # COMMENT(YT): Frobenius norm between the projected rdm.
    return beta * jnp.mean(jnp.array(
        [jnp.linalg.norm((rho_1 - rho_2), ord='fro') for rho_1, rho_2
            in zip(xz_estimates_model, xz_estimates)
        ])
    )
  return reg_fn  
