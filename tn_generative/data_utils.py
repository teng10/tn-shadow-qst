"""Helper functions for data loading and processing."""
import inspect

import numpy as np
import pandas as pd
from tn_generative import physical_systems


_PHYSICAL_SYSTEMS = {
    physical_systems.SurfaceCode.__name__: physical_systems.SurfaceCode,
    physical_systems.RubyRydbergPXP.__name__: physical_systems.RubyRydbergPXP,
}


def split_complex_ds(ds):
  """Split complex dataset variables into real and imaginary parts."""
  for var in ds.data_vars:
    if np.iscomplexobj(ds[var].values):
      ds[var + '_real'] = ds[var].real
      ds[var + '_imag'] = ds[var].imag
      ds = ds.drop_vars(var)
  return ds


def combine_complex_ds(ds):
  """Combine real and imaginary parts of complex dataset variables."""
  for var in ds.data_vars:
    if var.endswith('real'):
      var_real = var
      var_imag = var[:-len('_real')] + '_imag'
      ds[var[:-5]] = ds[var_real] + 1.j * ds[var_imag]
      ds = ds.drop_vars([var_real, var_imag])
  return ds


def merge_pd_tiled_config(df, config_df):
  """Merge pandas dataframe with tiled config dataframe."""
  tiled_config_df = pd.DataFrame(
      np.tile(config_df.to_numpy(), (df.index.stop, 1)),
      columns=config_df.columns)
  complete_df = pd.merge(
      df, tiled_config_df, left_index=True, right_index=True, how='outer')
  return complete_df


def physical_system_to_attrs_dict(physical_system):
  """Generates serializable dict representation of physical_system."""
  attrs_dict = {}
  args = inspect.signature(physical_system.__init__).parameters
  attrs_dict['physical_system_name'] = physical_system.__class__.__name__
  attrs_dict['physical_system_arg_names'] = ','.join(list(args.keys()))

  kwargs = {name: getattr(physical_system, name) for name in list(args.keys())}
  for k, v in kwargs.items():
    if np.issubdtype(type(v), np.integer):
      kwargs[k] = int(v)
    elif np.issubdtype(type(v), np.floating):
      kwargs[k] = float(v)
    elif isinstance(v, str):
      kwargs[k] = v
    else:
      raise ValueError(f'Unrecognized argument {type(v)=} for {k=}')
  attrs_dict.update(kwargs)
  return attrs_dict


def physical_system_from_attrs_dict(attrs_dict):
  """Construct physical system from attributes dictionary."""
  cls = _PHYSICAL_SYSTEMS[attrs_dict['physical_system_name']]
  kwargs = {k: attrs_dict[k]
            for k in attrs_dict['physical_system_arg_names'].split(',')
  }
  return cls(**kwargs)
