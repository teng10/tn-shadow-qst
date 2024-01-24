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
    delta,
    boundary,
 ) -> str:
  """Select the right dataset filename from available filenames."""
  # COMMENT: these lines are currently too long, but I don't know how to break.
  filenames = [
      'ruby_pxp_boundary_x_or_z_basis_sampler_size_x=4_size_y=2_d=60_delta=0.500_boundary_z_field=-0.6_boundary=periodic.nc',
      'ruby_pxp_boundary_x_or_z_basis_sampler_size_x=4_size_y=2_d=60_delta=1.700_boundary_z_field=-0.6_boundary=periodic.nc',
      'ruby_pxp_boundary_xz_basis_sampler_size_x=4_size_y=2_d=60_delta=0.500_boundary_z_field=-0.6_boundary=periodic.nc',
      'ruby_pxp_boundary_xz_basis_sampler_size_x=4_size_y=2_d=60_delta=1.700_boundary_z_field=-0.6_boundary=periodic.nc',
  ]
  unique_match = 0  # Check only one dataset is found.
  for name in filenames:
    regex = '_'.join([
        DEFAULT_TASK_NAME + '.*', # some file names have extra `boundary`
        sampler, f'{size_x=}', f'{size_y=}', 'd=(\d+)',
        f'{delta=:.3f}', '.*'+f'boundary={boundary}']
    ) + '.nc'
    if re.search(regex, name) is not None:
      unique_match += 1
      filename_match = name
  if unique_match != 1:
    raise ValueError(
        f'Found {unique_match} files matching {regex} while finding filename.'
    )
  return filename_match


def sweep_param_fn(
    sampler: str,
    size_x: int,
    size_y: int,
    boundary: str,
    delta: float,
    train_d: int,
    train_num_samples: int,
    train_beta: float,
    init_seed: int,
    reg_name: str,
    method: str,
) -> dict:
  """Helper function for formatting sweep parameters.

  Note: this function sweeps over dataset and training parameters.

  Args:
    sampler: dataset sampler name.
    size_x: dataset system size x.
    size_y: dataset system size y.
    boundary: dataset boundary condition: `periodic` or `open`.
    delta: dataset onsite z field delta.
    train_d: bond dimension.
    train_num_sample: number of training samples.
    train_beta: regularization strength.
    init_seed: random seed number for initializing mps.
    reg_name: regularization name.
    method: method for estimating regularization.

  Returns:
    dictionary of parameters for a single sweep.
  """
  if train_beta != 0. and reg_name == 'none':
    raise ValueError(f'Not meaningful {reg_name=} for {train_beta=}')
  return {
      'model.bond_dim': train_d,
      'data.num_training_samples': train_num_samples,
      'training.training_schemes.lbfgs_reg.reg_name': reg_name,
      'training.training_schemes.lbfgs_reg.reg_kwargs.beta': train_beta,
      'training.training_schemes.lbfgs_reg.reg_kwargs.method': method,
      'data.filename': get_dataset_name(
          sampler=sampler, size_x=size_x, size_y=size_y, delta=delta,
          boundary=boundary,
      ),
      'data.kwargs': {
          'task_name': DEFAULT_TASK_NAME,
          'sampler': sampler, 'size_x': size_x, 'size_y': size_y,
          'delta': delta, 'boundary': boundary,
      },
      'results.filename': '_'.join(['%JOB_ID', DEFAULT_TASK_NAME,
          sampler, f'{size_x=}', f'{size_y=}', f'{delta=:.3f}', f'boundary={boundary}',
          f'{train_d=}', f'{train_num_samples=}', f'{train_beta=:.3f}',
          f'{init_seed=}']
      ),
      'model.init_seed': init_seed,
  }


def sweep_nxm_ruby_fn(
    size_x: int,
    size_y: int,
    train_bond_dims: tuple[int],
    reg_name: str = 'hamiltonian',
    method: str = 'shadow',
    num_seeds: int = 10,
    train_samples: tuple[int] = (3_000, 7_000, 20_000, 40_000, 90_000),
    train_betas: tuple[float] = (0., 1., 5.),
    deltas: tuple[float] = (0., ),
    samplers: tuple[str] = (
        'x_y_z_basis_sampler', 'xz_basis_sampler', 'x_or_z_basis_sampler',
    ),
    boundary: str = 'periodic',
):
  for init_seed in range(num_seeds):
    for train_d in train_bond_dims:
      for sampler in samplers:
        for delta in deltas:
          for train_num_samples in train_samples:
            for train_beta in train_betas:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  boundary=boundary, delta=delta, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
                  method=method,
              )


# hexagon subsystems for ruby pxp.
subsystems = [
  [ 2,  4,  6, 13, 15, 23],
  [14, 16, 18, 25, 27, 35],
  [26, 28, 30, 37, 39, 47], 
  [8, 10, 0, 21, 19, 17], 
  [20, 22, 12, 33, 31, 29],
  [32, 34, 24, 45, 43, 41],
]
subsystem_kwargs = {'method':'explicit', 'explicit_subsystems': subsystems}

SWEEP_FN_REGISTRY = {
    'sweep_sc_4x2_fn_mps': list(sweep_nxm_ruby_fn(
        4, 2, train_bond_dims=(20, 40), reg_name='none',
        samplers=('x_or_z_basis_sampler', 'xz_basis_sampler', ),
        deltas=(1.7, 0.5), train_betas=(0., ),
    )),
    'sweep_sc_4x2_fn_shadow': list(sweep_nxm_ruby_fn(
        4, 2, train_bond_dims=(20, 10), reg_name='hamiltonian',
        method='shadow',
        samplers=('x_or_z_basis_sampler', ), # only randomized XZ.
        deltas=(1.7, ), train_betas=(0., 1., 5.),
    )),
    'sweep_sc_4x2_fn_xz_subsystem': list(sweep_nxm_ruby_fn(
        4, 2, train_bond_dims=(30, 20), reg_name='subsystem_xz_operators',
        method='shadow',
        samplers=('x_or_z_basis_sampler', ), # only randomized XZ.
        deltas=(0.5, ), train_betas=(0., 1., 5.),
    )),
    # TODO (YTZ): make this cleaner.
    'sweep_sc_4x2_fn_xz_subsystem_hexagon': [
        {**x, **{'training.training_schemes.lbfgs_reg.reg_kwargs.subsystem_kwargs': subsystem_kwargs}} for x in 
        list(sweep_nxm_ruby_fn(
        4, 2, train_bond_dims=(20, 10), reg_name='subsystem_xz_operators',
        method='shadow',
        samplers=('x_or_z_basis_sampler', ), # only randomized XZ.
        deltas=(1.7, ), train_betas=(1., 5.),
        ))
    ],    
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
  config.data.num_test_samples = 10_000
  # CLUSTER: needs to be changed
  config.data.dir = f'{HOME}/tn_shadow_dir/Data/{DEFAULT_TASK_NAME}'
  config.data.kwargs = {
      'task_name': DEFAULT_TASK_NAME,
      'sampler': 'xz_basis_sampler', 'size_x': 4, 'size_y': 2, 'd': 10,
      'delta': 1.7, 'boundary': 'open',
  }  # This is saved in the config dataframe.
  config.data.filename = '_'.join([
      '0', config.data.kwargs['task_name'],
      config.data.kwargs['sampler'], 'size_x=2', 'size_y=2',
      'd=20', 'delta=1.7000', 'open.nc']
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
  # minibatch pre-training config.
  minibatch_pretrain_config = config_dict.ConfigDict()
  minibatch_pretrain_config.training_scheme = 'minibatch'
  minibatch_pretrain_config.training_kwargs = {
      'batch_size': 256, 'record_loss_interval': 50
  }
  minibatch_pretrain_config.opt_kwargs = {'learning_rate': 1e-4}
  minibatch_pretrain_config.reg_name = 'none'
  # lbfgs training config.
  lbfgs_finetune_config = config_dict.ConfigDict()
  lbfgs_finetune_config.training_scheme = 'lbfgs'
  lbfgs_finetune_config.training_kwargs = {}
  lbfgs_finetune_config.reg_name = 'hamiltonian'
  lbfgs_finetune_config.reg_kwargs = {
      'beta': 0., 'method': 'mps', 'subsystem_kwargs': {'method': 'default'}
  }
  config.training.training_schemes = config_dict.ConfigDict()
  config.training.training_schemes.minibatch_no_reg = minibatch_pretrain_config
  config.training.training_schemes.lbfgs_reg = lbfgs_finetune_config
  # can be accessed via --config.training.training_schemes.
  # train through minibatch for 50 steps first, then lbfgs for 50 steps.
  config.training.training_sequence = ('minibatch_no_reg', 'lbfgs_reg')
  config.training.steps_sequence = (30000, 600)
  # Save options.
  config.results = config_dict.ConfigDict()
  config.results.save_results = True
  config.results.experiment_dir = ''.join([
      f'{HOME}/tn_shadow_dir/Results', '/Tests/%CURRENT_DATE/']
  )
  config.results.filename = '%JOB_ID_%TASK_ID'
  return config
