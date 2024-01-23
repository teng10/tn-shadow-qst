"""config file for data generation"""
import os
import re

from ml_collections import config_dict

HOME = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'surface_code'


def get_dataset_name(
    sampler: str,
    size_x: int,
    size_y: int,
    onsite_z_field: float,
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

      '2771105_surface_code_x_or_z_basis_sampler_size_x=3_size_y=5_d=10_onsite_z_field=0.000.nc',
      '2771117_surface_code_x_or_z_basis_sampler_size_x=3_size_y=7_d=20_onsite_z_field=0.000.nc',
      '2771128_surface_code_x_or_z_basis_sampler_size_x=3_size_y=9_d=40_onsite_z_field=0.000.nc',
      '2868588_surface_code_x_or_z_basis_sampler_size_x=3_size_y=11_d=80_onsite_z_field=0.000.nc',

      '3527909_surface_code_x_or_z_basis_sampler_size_x=11_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=11_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=13_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=13_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=15_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=15_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=3_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=5_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=5_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=7_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=7_size_y=3_d=5_onsite_z_field=0.100.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=9_size_y=3_d=5_onsite_z_field=0.000.nc',
      '3527909_surface_code_x_or_z_basis_sampler_size_x=9_size_y=3_d=5_onsite_z_field=0.100.nc',

      '6755832_surface_code_xz_basis_sampler_size_x=5_size_y=3_d=10_onsite_z_field=0.000.nc',
      '6755832_surface_code_xz_basis_sampler_size_x=7_size_y=3_d=10_onsite_z_field=0.000.nc',
      '6755832_surface_code_xz_basis_sampler_size_x=7_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6755832_surface_code_xz_basis_sampler_size_x=5_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6755832_surface_code_xz_basis_sampler_size_x=3_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=9_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=13_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=11_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=15_size_y=3_d=10_onsite_z_field=0.100.nc',
      '6755832_surface_code_xz_basis_sampler_size_x=9_size_y=3_d=10_onsite_z_field=0.000.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=15_size_y=3_d=10_onsite_z_field=0.000.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=11_size_y=3_d=10_onsite_z_field=0.000.nc',
      '6757142_surface_code_xz_basis_sampler_size_x=13_size_y=3_d=10_onsite_z_field=0.000.nc',

      '7069311_surface_code_x_or_z_basis_sampler_size_x=35_size_y=3_d=5_onsite_z_field=0.000.nc',
      '7069311_surface_code_xz_basis_sampler_size_x=35_size_y=3_d=5_onsite_z_field=0.000.nc',
      '7069311_surface_code_xz_basis_sampler_size_x=35_size_y=3_d=5_onsite_z_field=0.100.nc',
      '7069311_surface_code_x_or_z_basis_sampler_size_x=35_size_y=3_d=5_onsite_z_field=0.100.nc',
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
    sampler: str,
    size_x: int,
    size_y: int,
    onsite_z_field: float,
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
    onsite_z_field: dataset onsite z field.
    train_d: bond dimension.
    train_num_sample: number of training samples.
    train_beta: regularization strength.
    init_seed: random seed number for initializing mps.
    reg_name: name of regularization.
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


def surface_code_nxm_sweep_fn(
    size_x: int,
    size_y: int,
    train_bond_dims: tuple[int],
    method: str = 'mps',
    reg_name: str = 'hamiltonian',
    num_seeds: int = 10,
    train_samples: tuple[int] = (100, 500, 3_000, 20_000, 90_000),
    train_betas: tuple[float] = (0., 1., 5.),
    onsite_z_fields: tuple[float] = (0., ),
    samplers: tuple[str] = (
        'xz_basis_sampler', 'x_or_z_basis_sampler',
    ),
):
  for init_seed in range(num_seeds):
    for sampler in samplers:
      for onsite_z_field in onsite_z_fields:
        for train_d in train_bond_dims:
          for train_beta in train_betas:
            for train_num_samples in train_samples:
              yield sweep_param_fn(
                  sampler=sampler, size_x=size_x, size_y=size_y,
                  onsite_z_field=onsite_z_field, train_d=train_d,
                  train_num_samples=train_num_samples, train_beta=train_beta,
                  init_seed=init_seed,
                  reg_name=(reg_name if train_beta > 0 else 'none'),
                  method=method,
              )


SWEEP_FN_REGISTRY = {
    'sweep_sc_3x3_fn': list(
        surface_code_nxm_sweep_fn(3, 3, (10,), method='shadow', 
        num_seeds=10, onsite_z_fields=(0.1,), samplers=('x_or_z_basis_sampler',), 
        )
    ),
    'sweep_sc_5x5_fn': list(
        surface_code_nxm_sweep_fn(
            5, 5, (10, 20),
            samplers=('xz_basis_sampler', 'x_or_z_basis_sampler')),
    ),
    'sweep_sc_7x7_fn': list(
        surface_code_nxm_sweep_fn(
          7, 7, (20, 40),
          samplers=('x_or_z_basis_sampler', 'xz_basis_sampler')),
    ),
    # 'sweep_sc_3x5_fn': list(surface_code_nxm_sweep_fn(3, 5, (10, 20))),
    # 'sweep_sc_3x7_fn': list(surface_code_nxm_sweep_fn(3, 7, (20, 30))),
    # 'sweep_sc_3x9_fn': list(surface_code_nxm_sweep_fn(3, 9, (40, 50))),
    # 'sweep_sc_3x11_fn': list(surface_code_nxm_sweep_fn(3, 11, (80, 100))),
    'sweep_sc_size_y_3_fn': sum(
        [list(surface_code_nxm_sweep_fn(x, 3, [10],
            samplers=('x_or_z_basis_sampler', 'xz_basis_sampler', ), onsite_z_fields=(0., 0.1),
            num_seeds=5, )
        ) for x in [3, 5, 7, 9, 15]],
        start=[]
    ),
    'sweep_sc_size_y_3_shadow_fn': sum(
        [list(surface_code_nxm_sweep_fn(x, 3, [10],
            samplers=('x_or_z_basis_sampler', ), onsite_z_fields=(0., 0.1),
            num_seeds=10, method='shadow', )
        ) for x in [3, 5, 7, 9, 15]],
        start=[]
    ),    
    'sweep_sc_size_y_3_measurement_fn': sum(
        [list(surface_code_nxm_sweep_fn(x, 3, [10],
            samplers=('xz_basis_sampler', ), onsite_z_fields=(0., 0.1),
            num_seeds=10, method='measurement', )
        ) for x in [3, 5, 7, 9, 15]],
        start=[]
    ),        
    #TODO(YT): eventually generate this dataset.
    # 'sweep_sc_35x3_fn': list(surface_code_nxm_sweep_fn(35, 3, (5, ),
    # onsite_z_fields=(0., 0.1), )),
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
  # minibatch pre-training config.
  minibatch_pretrain_config = config_dict.ConfigDict()
  minibatch_pretrain_config.training_scheme = 'minibatch'
  minibatch_pretrain_config.training_kwargs = {
      'batch_size': 1024, 'record_loss_interval': 50
  }
  minibatch_pretrain_config.opt_kwargs = {'learning_rate': 1e-4}
  minibatch_pretrain_config.reg_name = 'none'
  # lbfgs training config.
  lbfgs_finetune_config = config_dict.ConfigDict()
  lbfgs_finetune_config.training_scheme = 'lbfgs'
  lbfgs_finetune_config.training_kwargs = {}
  lbfgs_finetune_config.reg_name = 'hamiltonian'
  lbfgs_finetune_config.reg_kwargs = {'beta': 0., 'method': 'mps'}
  #TODO(YT): consider add training schemes as dict.
  # training_schemes = {
  #     'minibatch_no_reg': minibatch_pretrain_config,
  #     'lbfgs_reg': lbfgs_finetune_config,
  # }
  # config.training.training_schemes = config_dict.ConfigDict(training_schemes)
  config.training.training_schemes = config_dict.ConfigDict()
  config.training.training_schemes.minibatch_no_reg = minibatch_pretrain_config
  config.training.training_schemes.lbfgs_reg = lbfgs_finetune_config
  # can be accessed via --config.training.training_schemes.
  # train through minibatch for 50 steps first, then lbfgs for 50 steps.
  config.training.training_sequence = ('minibatch_no_reg', 'lbfgs_reg')
  config.training.steps_sequence = (50000, 400)
  # Save options.
  config.results = config_dict.ConfigDict()
  config.results.save_results = True
  config.results.experiment_dir = ''.join([
      f'{HOME}/tn_shadow_dir/Results', '/Tests/%CURRENT_DATE/']
  )
  config.results.filename = '%JOB_ID_%TASK_ID'
  return config
