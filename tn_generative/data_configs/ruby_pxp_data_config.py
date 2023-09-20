"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')


DEFAULT_TASK_NAME = 'ruby_pxp'


def sweep_param_fn(size_x, size_y, d, delta, sampler):
  """Helper function for constructing sweep parameters."""
  return {
      'task.kwargs.size_x': size_x,
      'task.kwargs.size_y': size_y,
      'dmrg.bond_dims': d,
      'task.kwargs.delta': delta,
      'output.filename':  '_'.join(['%JOB_ID', DEFAULT_TASK_NAME, sampler,
          f'{size_x=}', f'{size_y=}', f'{d=}', f'{delta=:.3f}']),
  }


def sweep_sc_2x2_fn():
  # 2x2 unit cells sweep for ruby pxp
  size_x = 2
  size_y = 2
  for delta in np.arange(-10., 11., 10.):
    for d in [10, 20]:    
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x, size_y, d, delta, sampler)


SWEEP_FN_REGISTRY = {
    "sweep_sc_2x2_fn": list(sweep_sc_2x2_fn()),
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
  config.task.kwargs = {'size_x': 2, 'size_y': 2, 'delta': 0.}
  # sweep parameters.
  config.sweep_name = config_dict.placeholder(str)  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 20
  config.dmrg.solve_kwargs = {
      'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
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
