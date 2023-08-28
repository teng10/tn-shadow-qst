"""config file for data generation"""
import os
import re

from ml_collections import config_dict

HOME = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'surface_code'


def get_dataset_name(
    sampler,
    system_size,
    onsite_z_field
 ) -> str:
  """Select the right dataset filename from available filenames."""
  # COMMENT: these lines are currently too long, but I don't know how to break.
  filenames = [
      '66321312_surface_code_xz_basis_sampler_system_size=3_d=10_onsite_z_field=0.000.nc',
      '66321941_surface_code_xz_basis_sampler_system_size=5_d=40_onsite_z_field=0.000.nc',
      '66322999_surface_code_xz_basis_sampler_system_size=7_d=60_onsite_z_field=0.000.nc',
      '66324730_surface_code_x_or_z_basis_sampler_system_size=3_d=10_onsite_z_field=0.000.nc',
      '66325458_surface_code_x_or_z_basis_sampler_system_size=5_d=40_onsite_z_field=0.000.nc',
      '66325485_surface_code_x_or_z_basis_sampler_system_size=7_d=60_onsite_z_field=0.000.nc',
  ]
  unique_match = 0  # Check only one dataset is found.
  for name in filenames:
    regex = '_'.join([
        DEFAULT_TASK_NAME, sampler, f'{system_size=}', 'd=(\d+)',
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
    system_size,
    onsite_z_field,
    train_d,
    train_num_samples,
    train_beta,
    init_seed,
) -> dict:
  """Helper function for formatting sweep parameters.

  Note: this function sweeps over dataset and training parameters.

  Args:
    sampler: dataset sampler name.
    system_size: dataset system size.
    onsite_z_field: dataset onsite z field.
    train_d: bond dimension.
    train_num_sample: number of training samples.
    train_beta: regularization strength.
    init_seed: random seed number for initializing mps.

  Returns:
    dictionary of parameters for a single sweep.
  """
  return {
      'model.bond_dim': train_d,
      'data.num_training_samples': train_num_samples,
      'training.reg_kwargs.beta': train_beta,
      'data.filename': get_dataset_name(sampler, system_size, onsite_z_field),
      'data.kwargs': {
          'task_name': DEFAULT_TASK_NAME,
          'sampler': sampler, 'system_size': system_size,
          'onsite_z_field': onsite_z_field,
      },
      'results.filename': '_'.join(['%JOB_ID', DEFAULT_TASK_NAME,
          sampler, f'{system_size=}', f'{onsite_z_field=:.3f}',
          f'{train_d=}', f'{train_num_samples=}', f'{train_beta=:.3f}',
          f'{init_seed=}']
      ),
      'model.init_seed': init_seed,
  }


def sweep_sc_3x3_fn():
  # 5x5 sites surface code sweep
  for system_size in [3]:
    for init_seed in range(10):
      for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
        for onsite_z_field in [0.]:
          for train_d in [5, 10, 20]:
            for train_num_samples in [5000, 10_000, 30_000, 50_000, 100_000]:
              for train_beta in [0., 1., 5., 10.]:
                yield sweep_param_fn(
                    sampler=sampler, system_size=system_size,
                    onsite_z_field=onsite_z_field, train_d=train_d,
                    train_num_samples=train_num_samples, train_beta=train_beta,
                    init_seed=init_seed,
                )


def sweep_sc_5x5_fn():
  # 5x5 sites surface code sweep
  for system_size in [5]:
    for init_seed in range(10):
      for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
        for onsite_z_field in [0.]:
          for train_d in [10, 20, 40]:
            for train_num_samples in [5000, 10_000, 30_000, 50_000, 100_000]:
              for train_beta in [0., 1.]:
                yield sweep_param_fn(
                    sampler=sampler, system_size=system_size,
                    onsite_z_field=onsite_z_field, train_d=train_d,
                    train_num_samples=train_num_samples, train_beta=train_beta,
                    init_seed=init_seed,
                )


def sweep_sc_7x7_fn():
  # 7x7 sites surface code sweep
  for system_size in [7]:
    for init_seed in range(10):
      for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
        for onsite_z_field in [0.]:
          for train_d in [20, 40, 60]:
            for train_num_samples in [5000, 10_000, 30_000, 50_000, 100_000]:
              for train_beta in [0., 1.]:
                yield sweep_param_fn(
                    sampler=sampler, system_size=system_size,
                    onsite_z_field=onsite_z_field, train_d=train_d,
                    train_num_samples=train_num_samples, train_beta=train_beta,
                    init_seed=init_seed,
                )


SWEEP_FN_REGISTRY = {
    'sweep_sc_3x3_fn': list(sweep_sc_3x3_fn()),
    'sweep_sc_5x5_fn': list(sweep_sc_5x5_fn()),
    'sweep_sc_7x7_fn': list(sweep_sc_7x7_fn())
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
      'sampler': 'xz_basis_sampler', 'system_size': 3, 'd': 10,
      'onsite_z_field': 0.0
  }
  config.data.filename = '_'.join([
      '66321312', config.data.kwargs['task_name'], 
      config.data.kwargs['sampler'], 'system_size=3',
      'd=10', 'onsite_z_field=0.000.nc']
  )
  # Note: format of the data filename.
  # ['%JOB_ID', '%TASK_NAME', '%SAMPLER', '%SYSTEM_SIZE', '%D',
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
