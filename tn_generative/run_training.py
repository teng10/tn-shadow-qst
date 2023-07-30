"""Main file for running training."""
from absl import app
from absl import flags

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

config_flags.DEFINE_config_file('config')
FLAGS = flags.FLAGS

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY
REGULARIZER_REGISTRY = regularizers.REGULARIZER_REGISTRY


def run_full_batch_experiment(exp_config):
    train_config = exp_config.training
    model_config = exp_config.model
    ds = xr.open_dataset(exp_config.data.path)
    ds = data_utils.combine_complex_ds(ds)  # combine complex data into one ds.
    train_ds = ds.isel(sample=slice(0, exp_config.data.num_training_samples))
    system = TASK_REGISTRY[exp_config.task_name](**exp_config.task_kwargs)
    reg_fn = REGULARIZER_REGISTRY[train_config.reg_name]
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
    return complete_train_df, complete_eval_df, final_mps


def main(argv):
  config = FLAGS.config
  return run_full_batch_experiment(config)


if __name__ == '__main__':
  app.run(main)
  