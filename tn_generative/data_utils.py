"""Helper functions for data loading and processing."""
import numpy as np
import pandas as pd


def split_complex_ds(ds):
  for var in ds.data_vars:
    if np.iscomplexobj(ds[var].values):
      ds[var + '_real'] = ds[var].real
      ds[var + '_imag'] = ds[var].imag
      ds = ds.drop_vars(var)
  return ds


def combine_complex_ds(ds):
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
