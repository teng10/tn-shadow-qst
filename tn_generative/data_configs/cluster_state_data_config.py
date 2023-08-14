"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')


task_name = 'cluster_state'
sampling_method = 'x_or_z_basis_sampler'


def sweep_param_fn(system_size, d, onsite_z_field):
  """Helper function for constructing sweep parameters."""
  return {
      'task.kwargs.size_x': system_size,
      'task.kwargs.size_y': system_size,
      'dmrg.bond_dims': d,
      'task.kwargs.onsite_z_field': onsite_z_field,
      'output.filename':  '_'.join(['%JOB_ID', task_name, sampling_method,
          f'{system_size=}', f'{d=}', f'{onsite_z_field=:.3f}']),
  }


def sweep_sc_3x3_fn():
  # 3x3 sites surface code sweep
  for system_size in [3]:
    for d in [10, 20]:
      for onsite_z_field in np.linspace(0., 0.2, 11):
        yield sweep_param_fn(system_size, d, onsite_z_field)


def sweep_sc_5x5_fn():
  # 5x5 sites cluster state sweep
  for system_size in [5]:
    for d in [20, 40]:
      for onsite_z_field in np.linspace(0., 0.2, 11):
        yield sweep_param_fn(system_size, d, onsite_z_field)


def sweep_sc_7x7_fn():
  # 7x7 sites cluster state sweep
  for system_size in [7]:
    for d in [40, 60]:
      for onsite_z_field in np.linspace(0., 0.2, 11):
        yield sweep_param_fn(system_size, d, onsite_z_field)


sweep_fn_dict = {
    "sweep_sc_3x3_fn": list(sweep_sc_3x3_fn()),
    "sweep_sc_5x5_fn": list(sweep_sc_5x5_fn()),
    "sweep_sc_7x7_fn": list(sweep_sc_7x7_fn())
}


def get_config():
  """config for cluster state data generation."""
  config = config_dict.ConfigDict()
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = task_name
  config.task.kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field':0.}
  # sweep parameters.
  config.sweep_param = list(sweep_sc_3x3_fn())
  config.sweep_name = "sweep_sc_3x3_fn"  # Could change this in slurm script
  config.sweep_fn_dict = sweep_fn_dict  
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 20
  config.dmrg.solve_kwargs = {
      'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
  }
  # Sampler configuration.
  config.sampling = config_dict.ConfigDict()
  config.sampling.sampling_method = sampling_method
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