"""config file for data generation"""
from os import path

from ml_collections import config_dict

home = path.expanduser('~')

def get_config():
  config = config_dict.ConfigDict()
  config.model = config_dict.ConfigDict()
  config.model.bond_dim = 5
  config.model.dtype = 'complex128'
  config.model.init_seed = 43
  # data.
  config.data = config_dict.ConfigDict()
  config.data.num_training_samples = 1000
  config.data.name = '0727_data.nc'
  config.data.path = f'{home}/tn_shadow_dir/Data/Tests/'
  # training.
  config.training = config_dict.ConfigDict()
  config.training.num_training_steps = 10
  config.training.opt_kwargs = {}
  config.training.reg_name = 'hamiltonian'
  config.training.reg_kwargs = {'beta': 1., 'estimator': 'mps'}
  # physical system.
  config.task_name = 'surface_code'  #TODO(YT): consider loading from dataset. 
  # use zero field surface code to get only stabilizer MPOs.
  #TODO(YT): consider resetting onsite_z_field from dataset. 
  config.task_kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.}
  # Save options.
  config.output = config_dict.ConfigDict()
  config.output.save_data = True
  config.output.data_save_path = ''.join(
      [f'{home}/tn_shadow_dir/Data/Tests/', '%date_data']
  )
  return config
