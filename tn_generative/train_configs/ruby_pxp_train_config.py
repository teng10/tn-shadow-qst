"""config file for data generation"""
import os
import re

from ml_collections import config_dict

HOME = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'ruby_pxp'


def get_dataset_name(
    sampler,
    size_x,
    size_y,
    delta
 ) -> str:
  """Select the right dataset filename from available filenames."""
  # COMMENT: these lines are currently too long, but I don't know how to break.
  filenames = [
      '0_ruby_pxp_x_or_z_basis_sampler_size_x=2_size_y=2_d=20_delta=0.000.nc',
      '0_ruby_pxp_xz_basis_sampler_size_x=2_size_y=2_d=20_delta=0.000.nc',
  ]
  unique_match = 0  # Check only one dataset is found.
  for name in filenames:
    regex = '_'.join([
        DEFAULT_TASK_NAME, sampler, f'{size_x=}', f'{size_y=}', 'd=(\d+)',
        f'{delta=:.3f}.nc']
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
    delta,
    train_d,
    train_num_samples,
    train_beta,
    init_seed,
) -> dict:
  """Helper function for formatting sweep parameters.

  Note: this function sweeps over dataset and training parameters.

  Args:
    sampler: dataset sampler name.
    size_x: dataset system size x.
    size_y: dataset system size y.
    delta: dataset onsite z field delta.
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
      'data.filename': get_dataset_name(sampler, size_x, size_y, delta),
      'data.kwargs': {
          'task_name': DEFAULT_TASK_NAME,
          'sampler': sampler, 'size_x': size_x, 'size_y': size_y,
          'delta': delta,
      },
      'results.filename': '_'.join(['%JOB_ID', DEFAULT_TASK_NAME,
          sampler, f'{size_x=}', f'{size_y=}', f'{delta=:.3f}',
          f'{train_d=}', f'{train_num_samples=}', f'{train_beta=:.3f}',
          f'{init_seed=}']
      ),
      'model.init_seed': init_seed,
  }


def sweep_sc_2x2_fn():
  # 5x5 sites surface code sweep
  size_x = 2
  size_y = 2
  for init_seed in range(10):
    for sampler in ['xz_basis_sampler', 'x_or_z_basis_sampler']:
      for delta in [0.]:
        for train_d in [5, 10, 20]:
          for train_num_samples in [100, 1000, 5000, 10_000, 30_000, 50_000]:
            for train_beta in [0., 1.]:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  delta=delta, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
              )


SWEEP_FN_REGISTRY = {
    'sweep_sc_2x2_fn': list(sweep_sc_2x2_fn()),
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
      'sampler': 'xz_basis_sampler', 'size_x': 2, 'size_y': 2, 'd': 10,
      'delta': 0.0
  }
  config.data.filename = '_'.join([
      '0', config.data.kwargs['task_name'], 
      config.data.kwargs['sampler'], 'size_x=2', 'size_y=2',
      'd=20', 'delta=0.000.nc']
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