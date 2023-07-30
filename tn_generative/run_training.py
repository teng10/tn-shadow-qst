"""Main file for running training."""
from absl import app
from absl import flags
from datetime import datetime
import os
import pickle

from jax import config as jax_config
from ml_collections import config_flags
import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen
import pandas as pd
import xarray as xr

from tn_generative import data_generation
from tn_generative import data_utils
from tn_generative import train_utils
from tn_generative import regularizers
from tn_generative import types

config_flags.DEFINE_config_file('train_config')
FLAGS = flags.FLAGS

#TODO(YT): check whether can pass string detype and remove DTYPE_REGISTRY.
DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY
regularization = regularizers.REGULARIZER_REGISTRY


def run_full_batch_experiment(exp_config):
  current_date = datetime.now().strftime('%m%d')
  jax_config.update('jax_enable_x64', True)
  qtn.contraction.set_contract_backend('jax')
  train_config = exp_config.training
  model_config = exp_config.model
  ds = xr.open_dataset(exp_config.data.path)
  ds = data_utils.combine_complex_ds(ds)  # combine complex data into one ds.
  train_ds = ds.isel(sample=slice(0, exp_config.data.num_training_samples))
  #TODO(YT): better name for system?
  system = TASK_REGISTRY[exp_config.task_name](**exp_config.task_kwargs)
  reg_fn = regularization[train_config.reg_name]
  reg_fn = reg_fn(system=system, train_ds=train_ds, **train_config.reg_kwargs)
  qugen.rand.seed_rand(model_config.init_seed)
  model_mps = qtn.MPS_rand_state(
      train_ds.sizes['site'], model_config.bond_dim, dtype=model_config.dtype)
  train_df, eval_df, final_mps = train_utils.run_full_batch_training(
      model_mps, train_ds, train_config, reg_fn)

  # massaging configs to store all experiment parameters.
  config_df = pd.json_normalize(exp_config.to_dict(), sep='_')
  complete_eval_df = pd.merge(
      eval_df, config_df, left_index=True, right_index=True, how='outer')
  tiled_config_df = pd.DataFrame(
      np.tile(config_df.to_numpy(), (train_df.index.stop, 1)),
      columns=config_df.columns)
  complete_train_df = pd.merge(
      train_df, tiled_config_df,
      left_index=True, right_index=True, how='outer')
  if exp_config.output.save_data:
    if not os.path.exists(exp_config.output.filepath):
      os.makedirs(exp_config.output.filepath)
    save_path = exp_config.output.data_save_path.replace('%date', current_date)
    complete_train_df.to_pickle(save_path + '_train.p')
    complete_eval_df.to_pickle(save_path + '_eval.p')
    pickle.dump(final_mps, open(save_path + '_mps.p', 'wb'))
  return complete_train_df, complete_eval_df, final_mps


def main(argv):
  config = FLAGS.train_config
  return run_full_batch_experiment(config)


if __name__ == '__main__':
  app.run(main)
  