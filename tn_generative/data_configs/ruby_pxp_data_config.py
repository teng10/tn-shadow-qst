"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')


DEFAULT_TASK_NAME = 'ruby_pxp'


def sweep_param_fn(
    size_x: int,
    size_y: int,
    d: int,
    delta: float,
    boundary_z_field: float,
    sampler: str,
    boundary: str,
):
  """Helper function for constructing sweep parameters."""
  return {
      'task.kwargs.size_x': size_x,
      'task.kwargs.size_y': size_y,
      'task.kwargs.boundary_z_field': boundary_z_field,
      'task.kwargs.delta': delta,
      'task.kwargs.boundary': boundary,
      'dmrg.bond_dims': d,
      'sampling.sampling_method': sampler,
      'output.filename':  '_'.join(['%JOB_ID', DEFAULT_TASK_NAME, 'boundary',
          sampler, f'{size_x=}', f'{size_y=}', f'{d=}', f'{delta=:.3f}',
          f'{boundary_z_field=}', f'boundary={boundary}',
      ]),
  }


def sweep_sc_nxm_fn(
    size_x: int,
    size_y: int,
    deltas: np.ndarray,
    bond_dims: tuple[int],
    samplers: tuple[str] = (
        'xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler'
    ),
    boundary: str = 'periodic',
    boundary_z_field: float = -0.6,
):
  for delta in deltas:
    for d in bond_dims:
      for sampler in samplers:
        yield sweep_param_fn(
            size_x=size_x, size_y=size_y, d=d, boundary=boundary, delta=delta,
            boundary_z_field=boundary_z_field, sampler=sampler,
        )


SWEEP_FN_REGISTRY = {
    'sweep_sc_2x2_fn': list(sweep_sc_nxm_fn(
        size_x=2, size_y=2, deltas=np.arange(0.5, 2.5, 0.05), bond_dims=(20, 40)
    )),
    'sweep_sc_3x2_fn': list(sweep_sc_nxm_fn(
        size_x=3, size_y=2, deltas=np.arange(0.5, 2.5, 0.05), bond_dims=(20, 40)
    )),
    'sweep_sc_4x2_fn': list(sweep_sc_nxm_fn(
        size_x=4, size_y=2, deltas=np.arange(0.5, 2.5, 0.05), bond_dims=(20, 60)
    )),
    'sweep_sc_3x3_fn': list(sweep_sc_nxm_fn(
        size_x=3, size_y=3, deltas=np.arange(1.5, 1.85, 0.05), bond_dims=(40, 80)
    )),
    'sweep_sc_3x4_fn': list(sweep_sc_nxm_fn(
        size_x=3, size_y=4, deltas=np.arange(1.5, 1.85, 0.05), bond_dims=(80, 120)
    )),
}


def get_config():
  """Config dictionary."""
  config = config_dict.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.task_id = config_dict.placeholder(int)
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = DEFAULT_TASK_NAME
  config.task.kwargs = {
      'size_x': 2, 'size_y': 2, 'delta': 0., 'boundary': 'periodic',
      'boundary_z_field': -0.6,
  }
  # sweep parameters.
  config.sweep_name = config_dict.placeholder(str)  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 20
  config.dmrg.solve_kwargs = {
      'max_sweeps': 500, 'cutoffs': 1e-8, 'verbosity': 1,
      'sweep_sequence': 'RRLL',
  }
  # Sampler configuration.
  config.sampling = config_dict.ConfigDict()
  config.sampling.sampling_method = 'xz_basis_sampler'
  config.sampling.init_seed = 42
  config.sampling.num_samples = 100_000
  # Save options.
  config.output = config_dict.ConfigDict()
  config.output.save_data = True
  config.output.data_dir = f'{home}/tn_shadow_dir/Data/Tests/%CURRENT_DATE/'
  # by default we use date/job-id for file name.
  # need to keep this line for default value and jobs not on cluster.
  config.output.filename = '%JOB_ID'
  return config
