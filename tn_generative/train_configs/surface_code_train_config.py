"""config file for data generation"""
import os
import re

from ml_collections import config_dict

HOME = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'surface_code'


def get_dataset_name(
    sampler,
    size_x,
    size_y,
    onsite_z_field
 ) -> str:
  """Select the right dataset filename from available filenames."""
  # COMMENT: these lines are currently too long, but I don't know how to break.
  filenames = [
      '66321312_surface_code_xz_basis_sampler_size_x=3_size_y=3_d=10_onsite_z_field=0.000.nc',
      '66321941_surface_code_xz_basis_sampler_size_x=5_size_y=5_d=40_onsite_z_field=0.000.nc',
      '66322999_surface_code_xz_basis_sampler_size_x=7_size_y=7_d=60_onsite_z_field=0.000.nc',
      '66324730_surface_code_x_or_z_basis_sampler_size_x=3_size_y=3_d=10_onsite_z_field=0.000.nc',
      '66325458_surface_code_x_or_z_basis_sampler_size_x=5_size_y=5_d=40_onsite_z_field=0.000.nc',
      '66325485_surface_code_x_or_z_basis_sampler_size_x=7_size_y=7_d=60_onsite_z_field=0.000.nc',
      '2771105_surface_code_xz_basis_sampler_size_x=3_size_y=5_d=10_onsite_z_field=0.000.nc',
      '2771117_surface_code_xz_basis_sampler_size_x=3_size_y=7_d=20_onsite_z_field=0.000.nc',
      '2771128_surface_code_xz_basis_sampler_size_x=3_size_y=9_d=40_onsite_z_field=0.000.nc',
      '2868588_surface_code_xz_basis_sampler_size_x=3_size_y=11_d=80_onsite_z_field=0.000.nc',
      '2771105_surface_code_x_or_z_basis_sampler_size_x=3_size_y=5_d=10_onsite_z_field=0.000.nc',
      '2771117_surface_code_x_or_z_basis_sampler_size_x=3_size_y=7_d=20_onsite_z_field=0.000.nc',
      '2771128_surface_code_x_or_z_basis_sampler_size_x=3_size_y=9_d=40_onsite_z_field=0.000.nc',
      '2868588_surface_code_x_or_z_basis_sampler_size_x=3_size_y=11_d=80_onsite_z_field=0.000.nc',
      '2867669_surface_code_x_y_z_basis_sampler_size_x=3_size_y=3_d=10_onsite_z_field=0.000.nc',
      '2771105_surface_code_x_y_z_basis_sampler_size_x=3_size_y=5_d=10_onsite_z_field=0.000.nc',
      '2771117_surface_code_x_y_z_basis_sampler_size_x=3_size_y=7_d=20_onsite_z_field=0.000.nc',
      '2771128_surface_code_x_y_z_basis_sampler_size_x=3_size_y=9_d=40_onsite_z_field=0.000.nc',
      '2868588_surface_code_x_y_z_basis_sampler_size_x=3_size_y=11_d=80_onsite_z_field=0.000.nc',      

            
  ]
  unique_match = 0  # Check only one dataset is found.
  for name in filenames:
    regex = '_'.join([
        DEFAULT_TASK_NAME, sampler, f'{size_x=}', f'{size_y=}', 'd=(\d+)',
        f'{onsite_z_field=:.3f}.nc']
    )
    if re.search(regex, name) is not None:
      unique_match += 1
      filename_match = name
  if unique_match != 1:
    raise ValueError(
        f'Found {unique_match} files matching {regex} while finding filename.'
    )
  return filename_match


def sweep_param_fn(
    sampler,
    size_x,
    size_y,
    onsite_z_field,
    train_d,
    train_num_samples,
    train_beta,
    init_seed,
    reg_name,
) -> dict:
  """Helper function for formatting sweep parameters.

  Note: this function sweeps over dataset and training parameters.

  Args:
    sampler: dataset sampler name.
    size_x: dataset system size x.
    size_y: dataset system size y.
    onsite_z_field: dataset onsite z field.
    train_d: bond dimension.
    train_num_sample: number of training samples.
    train_beta: regularization strength.
    init_seed: random seed number for initializing mps.

  Returns:
    dictionary of parameters for a single sweep.
  """
  if train_beta != 0. and reg_name == 'none':
    raise ValueError(f'Not meaningful {reg_name=} for {train_beta=}')  
  return {
      'model.bond_dim': train_d,
      'data.num_training_samples': train_num_samples,
      'training.reg_name': reg_name,
      'training.reg_kwargs.beta': train_beta,
      'data.filename': get_dataset_name(sampler, size_x, size_y, onsite_z_field),
      'data.kwargs': {
          'task_name': DEFAULT_TASK_NAME,
          'sampler': sampler, 'size_x': size_x, 'size_y': size_y,
          'onsite_z_field': onsite_z_field,
      },      
      'results.filename': '_'.join(['%JOB_ID', DEFAULT_TASK_NAME,
          sampler, f'{size_x=}', f'{size_y=}', f'{onsite_z_field=:.3f}',
          f'{train_d=}', f'{train_num_samples=}', f'{train_beta=:.3f}',
          f'{init_seed=}']
      ),
      'model.init_seed': init_seed,
  }


def sweep_sc_3x3_fn():
  size_x = 3
  size_y = 3
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [5, 10]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1., 5.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_5x5_fn():
  size_x = 5
  size_y = 5
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [10, 20, 40]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_7x7_fn():
  size_x = 7
  size_y = 7
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [20, 40, 60]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_3x5_fn():
  size_x = 3
  size_y = 5
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [10, 20]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1., 5.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_3x7_fn():
  size_x = 3
  size_y = 7
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [20, 30]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1., 5.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_3x9_fn():
  size_x = 3
  size_y = 9
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [40, 50]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1., 5.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


def sweep_sc_3x11_fn():
  size_x = 3
  size_y = 11
  reg_name = 'hamiltonian'
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler', 'x_y_z_basis_sampler']:
      for onsite_z_field in [0.]:
        for train_d in [80, 100]:
          for train_num_samples in [100, 500, 3_000, 20_000, 100_000]:
            for train_beta in [0., 1., 5.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
              )


SWEEP_FN_REGISTRY = {
    'sweep_sc_3x3_fn': list(sweep_sc_3x3_fn()),
    'sweep_sc_5x5_fn': list(sweep_sc_5x5_fn()),
    'sweep_sc_7x7_fn': list(sweep_sc_7x7_fn()),
    'sweep_sc_3x5_fn': list(sweep_sc_3x5_fn()),
    'sweep_sc_3x7_fn': list(sweep_sc_3x7_fn()),
    'sweep_sc_3x9_fn': list(sweep_sc_3x9_fn()),
    'sweep_sc_3x11_fn': list(sweep_sc_3x11_fn()),
}


def get_config():
  config = config_dict.ConfigDict()
  # job properties.
  config.job_id = config_dict.placeholder(int)
  config.task_id = config_dict.placeholder(int)
  # model.
  config.model = config_dict.ConfigDict()
  config.model.bond_dim = 5
  config.model.dtype = 'complex128'
  config.model.init_seed = 43
  # data.
  config.data = config_dict.ConfigDict()
  config.data.num_training_samples = 1000
  # CLUSTER: needs to be changed
  config.data.dir = f'{HOME}/tn_shadow_dir/Data/{DEFAULT_TASK_NAME}'
  config.data.kwargs = {
      'task_name': DEFAULT_TASK_NAME,
      'sampler': 'xz_basis_sampler', 'size_x': 3, 'size_y':3, 'd': 10,
      'onsite_z_field': 0.0
  }  #TODO(YT): this can be loaded from xr dataset attrs to reduce bugs.
  config.data.filename = '_'.join([
      '66321312', config.data.kwargs['task_name'], 
      config.data.kwargs['sampler'], 'size_x=3', 'size_y=3',
      'd=10', 'onsite_z_field=0.000.nc']
  )
  # Note: format of the data filename.
  # ['%JOB_ID', '%TASK_NAME', '%SAMPLER', '%SIZE_X', '%SIZE_Y', '%D',
  # '%ONSITE_Z_FIELD']
  # sweep parameters.
  # CLUSTER: could change this in slurm script
  config.sweep_name = config_dict.placeholder(str)
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # training.
  config.training = config_dict.ConfigDict()
  config.training.num_training_steps = 200
  config.training.opt_kwargs = {}
  config.training.reg_name = 'hamiltonian'
  config.training.reg_kwargs = {'beta': 0., 'estimator': 'mps'}
  # Save options.
  config.results = config_dict.ConfigDict()
  config.results.save_results = True
  config.results.experiment_dir = ''.join([
      f'{HOME}/tn_shadow_dir/Results', '/Tests/%CURRENT_DATE/']
  )
  config.results.filename = '%JOB_ID_%TASK_ID'
  return config
