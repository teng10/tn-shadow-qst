"""Main file for running training."""
# How to run this file:
# python -m tn_generative.run_training \
# --train_config=tn_generative/train_configs/surface_code_training_config.py \
# --train_config.job_id=0828 \
# --train_config.task_id=0 \
# --train_config.sweep_name="sweep_sc_3x3_fn" \
# --train_config.training.num_training_steps=20
from absl import app
from absl import flags
from datetime import datetime
import os
import logging

from jax import config as jax_config
from ml_collections import config_flags
import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen
import pandas as pd
import xarray as xr

from tn_generative import data_generation
from tn_generative import data_utils
from tn_generative import mps_utils
from tn_generative import train_utils
from tn_generative import regularizers
from tn_generative import types

config_flags.DEFINE_config_file('train_config')
FLAGS = flags.FLAGS

#TODO(YT): check whether can pass string detype and remove DTYPE_REGISTRY.
DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY
regularization = regularizers.REGULARIZER_REGISTRY


def run_full_batch_experiment(config):
  current_date = datetime.now().strftime('%m%d')
  jax_config.update('jax_enable_x64', True)
  qtn.contraction.set_contract_backend('jax')
  if config.sweep_name in config.sweep_fn_registry:
    sweep_params = config.sweep_fn_registry[config.sweep_name]
    config.update_from_flattened_dict(sweep_params[config.task_id])
  elif config.sweep_name is None:
    pass
  else:
    raise ValueError(f'Invalid sweep name {config.sweep_name}')
  # TODO(YT): temporary solution to delete the sweep_fn. Find a better way!
  config.unlock()
  del config.sweep_fn_registry
  config.lock()
  train_config = config.training
  model_config = config.model
  datapath = os.path.join(config.data.dir, config.data.filename)
  logging.info(f'Using {datapath=}')
  ds = xr.open_dataset(datapath)
  # Load dataset by combining real&imag fields into complex fields.
  ds = data_utils.combine_complex_ds(ds)
  train_ds = ds.isel(sample=slice(0, config.data.num_training_samples))
  # TODO(YT): better way to reload attrs from dataset.
  # Add utils for physical_system: `to_xarray_attrs`, `from_xarray_attrs`.
  ds_attrs = ds.attrs.copy()
  ds_attrs.pop('name')
  physical_system = TASK_REGISTRY[ds.name](**ds_attrs)
  if train_config.reg_kwargs.beta != 0.:
    reg_fn = regularization[train_config.reg_name]
  else:
    reg_fn = None
  if reg_fn is not None:
    reg_fn = reg_fn(
        system=physical_system, train_ds=train_ds,
        **train_config.reg_kwargs
    )
  qugen.rand.seed_rand(model_config.init_seed)
  model_mps = qtn.MPS_rand_state(
      train_ds.sizes['site'], model_config.bond_dim, dtype=model_config.dtype)
  train_df, eval_df, final_mps = train_utils.run_full_batch_training(
      model_mps, train_ds, train_config, reg_fn)

  # massaging configs to store all experiment parameters.
  config_df = pd.json_normalize(config.to_dict(), sep='_')
  complete_eval_df = pd.merge(
      eval_df, config_df, left_index=True, right_index=True, how='outer')
  tiled_config_df = pd.DataFrame(
      np.tile(config_df.to_numpy(), (train_df.index.stop, 1)),
      columns=config_df.columns)
  complete_train_df = pd.merge(
      train_df, tiled_config_df,
      left_index=True, right_index=True, how='outer')
  if config.results.save_results:
    results_dir = config.results.experiment_dir.replace(
        '%CURRENT_DATE', current_date
    )
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    results_filename = config.results.filename.replace(
        '%JOB_ID', str(config.job_id)
    )
    save_path = os.path.join(results_dir, results_filename)
    complete_train_df.to_csv(save_path + '_train.csv')
    complete_eval_df.to_csv(save_path + '_eval.csv')
    mps_ds = data_utils.split_complex_ds(mps_utils.mps_to_xarray(final_mps))
    mps_ds.to_netcdf(save_path + '_mps.nc')
  return complete_train_df, complete_eval_df, final_mps


def main(argv):
  config = FLAGS.train_config
  return run_full_batch_experiment(config)


if __name__ == '__main__':
  app.run(main)
