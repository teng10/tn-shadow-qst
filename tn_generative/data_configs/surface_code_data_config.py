"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'surface_code'


def sweep_param_fn(size_x, size_y, d, onsite_z_field, sampler):
  """Helper function for constructing sweep parameters."""
  return {
      'task.kwargs.size_x': size_x,
      'task.kwargs.size_y': size_y,
      'dmrg.bond_dims': d,
      'task.kwargs.onsite_z_field': onsite_z_field,
      'output.filename':  '_'.join(['%JOB_ID', DEFAULT_TASK_NAME, sampler,
          f'{size_x=}', f'{size_y=}', f'{d=}', f'{onsite_z_field=:.3f}']),
  }


def sweep_sc_7x7_fn():
  """Sweep over surface code data configs."""
  size_x = 7
  size_y = 7
  for d in [40, 60]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_5x5_fn():
  """Sweep over surface code data configs."""
  size_x = 5
  size_y = 5
  for d in [20, 40]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_3x3_fn():
  """Sweep over surface code data configs."""
  size_x = 3
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_3x5_fn():
  """Sweep over surface code data configs."""
  size_x = 3
  size_y = 5
  for d in [10, 20]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_3x7_fn():
  """Sweep over surface code data configs."""
  size_x = 3
  size_y = 7
  for d in [10, 20]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_3x9_fn():
  """Sweep over surface code data configs."""
  size_x = 3
  size_y = 9
  for d in [20, 40]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_3x11_fn():
  """Sweep over surface code data configs."""
  size_x = 3
  size_y = 11
  for d in [40, 60]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_5x3_fn():
  """Sweep over surface code data configs."""
  size_x = 5
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_7x3_fn():
  """Sweep over surface code data configs."""
  size_x = 7
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_9x3_fn():
  """Sweep over surface code data configs."""
  size_x = 9
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_11x3_fn():
  """Sweep over surface code data configs."""
  size_x = 11
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_13x3_fn():
  """Sweep over surface code data configs."""
  size_x = 13
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


def sweep_sc_15x3_fn():
  """Sweep over surface code data configs."""
  size_x = 15
  size_y = 3
  for d in [5, 10]:
    for onsite_z_field in np.linspace(0., 0.1, 2):
      for sampler in [
          'xz_basis_sampler', 'x_or_z_basis_sampler',
          'x_y_z_basis_sampler',
      ]:
        yield sweep_param_fn(size_x=size_x, size_y=size_y, d=d,
            onsite_z_field=onsite_z_field, sampler=sampler
        )


SWEEP_FN_REGISTRY = {
    "sweep_sc_3x3_fn": list(sweep_sc_3x3_fn()),
    "sweep_sc_5x5_fn": list(sweep_sc_5x5_fn()),
    "sweep_sc_7x7_fn": list(sweep_sc_7x7_fn()), 
    "sweep_sc_3x5_fn": list(sweep_sc_3x5_fn()),
    "sweep_sc_3x7_fn": list(sweep_sc_3x7_fn()),
    "sweep_sc_3x9_fn": list(sweep_sc_3x9_fn()),
    "sweep_sc_3x11_fn": list(sweep_sc_3x11_fn()),
    "sweep_sc_size_y_3_fn": list(sweep_sc_3x3_fn) + list(sweep_sc_5x3_fn()) + \
        list(sweep_sc_7x3_fn()) + list(sweep_sc_9x3_fn()) + \
        list(sweep_sc_11x3_fn()) + list(sweep_sc_13x3_fn()) + \
        list(sweep_sc_15x3_fn()),
}


def get_config():
  """config using surface code as an example."""
  config = config_dict.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.task_id = config_dict.placeholder(int)
  # Task configuration.
  config.dtype = 'complex128'
  config.task = config_dict.ConfigDict()
  config.task.name = DEFAULT_TASK_NAME
  config.task.kwargs = {'size_x': 5, 'size_y': 5, 'onsite_z_field': 0.1}
  # sweep parameters.
  config.sweep_name = "sweep_sc_3x3_fn"  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # DMRG configuration.
  config.dmrg = config_dict.ConfigDict()
  config.dmrg.bond_dims = 10
  config.dmrg.solve_kwargs = {
      'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
  }
  # Sampler configuration.
  config.sampling = config_dict.ConfigDict()
  config.sampling.sampling_method = 'x_or_z_basis_sampler'
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
