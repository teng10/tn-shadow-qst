"""Helper functions for data loading and processing."""
import numpy as np


def split_complex_ds(ds):
  for var in ds.data_vars:
    if np.iscomplexobj(ds[var].values):
      ds[var + "_real"] = ds[var].real
      ds[var + "_imag"] = ds[var].imag
      ds = ds.drop_vars(var)
  return ds

def combine_complex_ds(ds):
  for var in ds.data_vars:
    if "_real" in var:
      var_real = var
      var_imag = var[:-5] + "_imag"
      ds[var[:-5]] = ds[var_real] + 1.j * ds[var_imag]
      ds = ds.drop_vars([var_real, var_imag])
  return ds
