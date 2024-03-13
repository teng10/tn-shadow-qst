"""Helper functions for data loading and processing."""
import functools
import math
import inspect

import numpy as np
import pandas as pd
import xarray as xr

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


def make_iterator_over_ds(ds, dim, batch_size):
  """Creates iterator over `ds` with `batch_size` slices along `dim`."""
  buckets = math.ceil(ds.sizes[dim] / batch_size)
  for i in range(buckets):
    yield ds.isel({dim: slice(i * batch_size, (i+1) * batch_size)})


def stream_mean_over_dim(ds, fn, dim, batch_size=500):
  """Compute the streamed mean of `fn` applied to `ds` along `dim`."""
  def _stream_mean_fn(c, x):
    # c is carry of the form (mean, count)
    # x is the next slice of the dataset
    mean, count = c
    new_count = x.sizes[dim]
    return (
        mean * count / (count + new_count) +
        fn(x).mean(dim=dim) * new_count / (count + new_count), count + new_count
    )

  ds_iterator = make_iterator_over_ds(ds, dim, batch_size)
  return functools.reduce(_stream_mean_fn, ds_iterator, (0, 0))[0]


def merge_pd_tiled_config(df, config_df):
  """Merge pandas dataframe with tiled config dataframe."""
  tiled_config_df = pd.DataFrame(
      np.tile(config_df.to_numpy(), (df.index.stop, 1)),
      columns=config_df.columns)
  complete_df = pd.merge(
      df, tiled_config_df, left_index=True, right_index=True, how='outer')
  return complete_df


# TODO(YT): move to observables_utils.py
def compute_onsite_pauli_expectations(mps, physical_system):
  """Compute the expectation value of the onsite Pauli operators.

  Args:
    mps: an MPS to compute properties of.
    physical_system: A task system.

  Returns:
    onsite_z_expectation: list of expectation values of onsite Pauli Z.
    onsite_x_expectation: list of expectation values of onsite Pauli X.
    onsite_y_expectation: list of expectation values of onsite Pauli Y.
  """
  # COMMENT: could write dmrg_analysis_utils instead.
  onsite_z_terms = [(1., ('z', i)) for i in range(physical_system.n_sites)]
  onsite_z_mpos = physical_system.get_obs_mpos(onsite_z_terms)
  onsite_z_expectation = np.array(
      [(mps.H @ (s.apply(mps))) for s in onsite_z_mpos]
  )
  onsite_x_terms = [(1., ('x', i)) for i in range(physical_system.n_sites)]
  onsite_x_mpos = physical_system.get_obs_mpos(onsite_x_terms)
  onsite_x_expectation = np.array(
      [(mps.H @ (s.apply(mps))) for s in onsite_x_mpos]
  )
  onsite_y_terms = [(1., ('y', i)) for i in range(physical_system.n_sites)]
  onsite_y_mpos = physical_system.get_obs_mpos(onsite_y_terms)
  onsite_y_expectation = np.array(
      [(mps.H @ (s.apply(mps))) for s in onsite_y_mpos]
  )
  return xr.Dataset(
      {
          'onsite_z_ev': (['site'], onsite_z_expectation),
          'onsite_x_ev': (['site'], onsite_x_expectation),
          'onsite_y_ev': (['site'], onsite_y_expectation),
      },
      coords={
          'site': np.arange(physical_system.n_sites),
      }
  )


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


def _discard_odd_basis_measurements(
    ds: xr.Dataset,
    basis: int=1,
    mod: int=2,
) -> xr.Dataset:
  """Discard measurements for `basis` occuring with nonzero `mod` frequency.
  
  Example: to throw away any pauli x measurements, 
  set `basis=0` and `mod=mps.L+1`.
  Args:
    ds: The dataset to discard the basis measurement from.
    basis: The basis to discard. Default (1) is y basis.
    mod: The modulus of the number of measurements to discard. Default is 2.

  Returns:
    The dataset with the basis measurement discarded.
  """
  def _mask_fn(array):
    return np.count_nonzero(array == basis, axis=-1) % mod == 0
  
  mask = xr.apply_ufunc(
      _mask_fn, ds.basis,
      input_core_dims=[['site']], output_core_dims=[[]],
  ) # axis=-1 because of the way xr handles broadcasting with CORE_DIMENSIONS
  return ds.where(mask, drop=True).astype(int)
