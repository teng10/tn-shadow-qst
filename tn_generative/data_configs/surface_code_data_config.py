"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')


def sweep_sc_fn(size_x, size_y):
  """Helper function for sweeping over surface code configs."""
  # 5x5 sites surface code sweep
  if size_x == 5 and size_y == 5:
    for d in [5, 10, 20]:
      for onsite_z_field in np.linspace(0., 0.5, 11):
        for num_samples in [100, 1000, 5000, 10_000, 15_000, 20_000]:
          yield {'dmrg.bond_dims': d,
                 'task.kwargs.onsite_z_field': onsite_z_field,
                 'sampling.num_samples': num_samples,
                 }

  # 3x3 sites surface code sweep
  if size_x == 3 and size_y == 3:
    for d in [5, 10]:
      for onsite_z_field in np.linspace(0., 0.2, 4):
        for num_samples in [100, 1000]:
          yield {'dmrg.bond_dims': d,
                 'task.kwargs.onsite_z_field': onsite_z_field,
                 'sampling.num_samples': num_samples,
                 }          
      

def get_config():
  """config using surface code as an example."""
  config = config_dict.ConfigDict()
  # job properties
  config.job_id = 731 # config_dict.placeholder(int)
  config.task_id = 0 # config_dict.placeholder(int)  
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = 'surface_code'
  config.task.kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.1}
  # sweep parameters.
  config.sweep_param = list(
      sweep_sc_fn(3, 3)
  )
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 5
  config.dmrg.solve_kwargs = {
      'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
  }
  # Sampler configuration.
  config.sampling = config_dict.ConfigDict()
  config.sampling.sampling_method = 'xz_basis_sampler'
  config.sampling.init_seed = 42
  config.sampling.num_samples = 500
  # Save options.
  config.output = config_dict.ConfigDict()
  config.output.save_data = True
  config.output.data_dir = f'{home}/tn_shadow_dir/Data/Tests/'
  config.output.filename = 'data_task%d'
  return config
