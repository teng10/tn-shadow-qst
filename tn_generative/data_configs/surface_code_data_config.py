"""config file for data generation."""
from os.path import expanduser

from ml_collections import config_dict

home = expanduser('~')

def get_config():
  """config using surface code as an example."""
  config = config_dict.ConfigDict()
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = 'surface_code'
  config.task.kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.1}
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
  config.output.data_save_path = f'{home}/tn_shadow_dir/Data/Tests/'
  config.output.filename = 'data'
  return config
