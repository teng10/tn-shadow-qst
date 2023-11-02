"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'surface_code'


def sweep_param_fn(
    size_x: int,
    size_y: int,
    d: int,
    onsite_z_field: float,
    sampler: str,
) -> dict:
  """Helper function for constructing dmrg sweep parameters."""
  return {
      'task.kwargs.size_x': size_x,
      'task.kwargs.size_y': size_y,
      'dmrg.bond_dims': d,
      'sampling.sampling_method': sampler,
      'task.kwargs.onsite_z_field': onsite_z_field,
      'output.filename':  '_'.join(['%JOB_ID', DEFAULT_TASK_NAME, sampler,
          f'{size_x=}', f'{size_y=}', f'{d=}', f'{onsite_z_field=:.3f}']),
  }


def surface_code_nxm_sweep_fn(
    size_x: int,
    size_y: int,
    bond_dims: list[int],
    onsite_z_fields = np.linspace(0., 0.1, 2),
    samplers: tuple[str] = (
        'xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler'
    ),
):
  """Sweep over surface code data configs."""
  for d in bond_dims:
    for onsite_z_field in onsite_z_fields:
      for sampler in samplers:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


SWEEP_FN_REGISTRY = {
    'sweep_sc_3x3_fn': list(surface_code_nxm_sweep_fn(3, 3, [5, 10])),
    'sweep_sc_5x5_fn': list(surface_code_nxm_sweep_fn(5, 5, [20, 40])),
    'sweep_sc_7x7_fn': list(surface_code_nxm_sweep_fn(7, 7, [40, 60])),
    'sweep_sc_3x5_fn': list(surface_code_nxm_sweep_fn(3, 5, [10, 20])),
    'sweep_sc_3x7_fn': list(surface_code_nxm_sweep_fn(3, 7, [10, 20])),
    'sweep_sc_3x9_fn': list(surface_code_nxm_sweep_fn(3, 9, [20, 40])),
    'sweep_sc_3x11_fn': list(surface_code_nxm_sweep_fn(3, 11, [40, 60])),
    'sweep_sc_size_y_3_fn': sum(
        [list(
            surface_code_nxm_sweep_fn(x, 3, [5, 10])
        ) for x in [3, 5, 7, 9, 11, 13, 15]],
        start=[]
    ),
    'sweep_sc_33x3_fn': list(surface_code_nxm_sweep_fn(33, 3, [5, 10])),
}


def get_config():
  """config using surface code as an example."""
  config = config_dict.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.task_id = config_dict.placeholder(int)
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = DEFAULT_TASK_NAME
  config.task.kwargs = {'size_x': 5, 'size_y': 5, 'onsite_z_field': 0.1}
  # sweep parameters.
  config.sweep_name = 'sweep_sc_3x3_fn'  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 10
  config.dmrg.solve_kwargs = {
      'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
  }
  # Sampler configuration.
  config.sampling = config_dict.ConfigDict()
  config.sampling.sampling_method = 'x_or_z_basis_sampler'
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
