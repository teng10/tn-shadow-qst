"""Main file for running training."""
# How to run this file using command line:
# python -m tn_generative.run_training \
# --train_config=tn_generative/train_configs/surface_code_train_config.py \
# --train_config.job_id=0530 \
# --train_config.task_id=0 \
# --train_config.data.dir=tn_generative/test_data/ \
# --train_config.data.filename=surface_code_xz.nc \
# --train_config.results.experiment_dir=./ \
# --train_config.training.steps_sequence="(5000,400)" \
# --train_config.data.num_training_samples=10000 \
# --train_config.model.bond_dim=10
# How to run this file using config file:
# python -m tn_generative.run_training \
# --train_config=tn_generative/train_configs/surface_code_train_config.py \
# --train_config.job_id=0828 \
# --train_config.task_id=0 \
# --train_config.sweep_name="sweep_sc_3x3_fn" \
# --train_config.training.steps_sequence="(5000,400)" \
from absl import app
from absl import flags
from datetime import datetime
import os
import logging
from typing import Any, Dict, Tuple

from jax import config as jax_config
from ml_collections import config_flags
import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen
import pandas as pd
import xarray as xr

from tn_generative import data_utils
from tn_generative import mps_utils
from tn_generative import train_utils
from tn_generative import types

config_flags.DEFINE_config_file('train_config')
FLAGS = flags.FLAGS

#TODO(YT): check whether can pass string detype and remove DTYPE_REGISTRY.
DTYPES_REGISTRY = types.DTYPES_REGISTRY
TRAIN_SCHEME_REGISTRY = train_utils.TRAIN_SCHEME_REGISTRY


def run_full_batch_experiment(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, qtn.MatrixProductState]]:
  current_date = datetime.now().strftime('%m%d')
  jax_config.update('jax_enable_x64', True)
  qtn.contraction.set_contract_backend('jax')
  if config.sweep_name in config.sweep_fn_registry:
    sweep_params = config.sweep_fn_registry[config.sweep_name]
    config.update_from_flattened_dict(sweep_params[config.task_id])
    logging.info(f'Updating configs using {config.sweep_name=}')
  elif config.sweep_name == None:
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
  # partition the dataset into training and test.
  test_ds = ds.isel(sample=slice(-config.data.num_test_samples, None))
  if (config.data.num_training_samples + config.data.num_test_samples
      > ds.sizes['sample']
  ):
    raise ValueError(
        f'Train samples {config.data.num_training_samples} + test samples \
        {config.data.num_test_samples} > total samples {ds.sizes["sample"]}.'
    )
  qugen.rand.seed_rand(model_config.init_seed)
  model_mps = qtn.MPS_rand_state(
      train_ds.sizes['site'], model_config.bond_dim, dtype=model_config.dtype)
  # Define training scheme.
  train_dfs = []
  eval_dfs = []
  mps_sequences = {}
  for i, (schedule_step, train_schedule_name) in enumerate(
      zip(
          train_config.steps_sequence, train_config.training_sequence
      )
  ):
    train_scheme_config = train_config.training_schemes[train_schedule_name]
    train_scheme = TRAIN_SCHEME_REGISTRY[train_scheme_config.training_scheme]
    train_df, eval_df, model_mps = train_scheme(
        model_mps, train_ds, test_ds, train_scheme_config, schedule_step
    )  # Choose to leave step as a separate argument from the config.
    current_sequence = '_'.join([train_scheme_config.training_scheme, str(i)])
    train_df['current_sequence'] = current_sequence
    eval_df['current_sequence'] = current_sequence
    train_dfs.append(train_df)
    eval_dfs.append(eval_df)
    mps_sequences[current_sequence] = model_mps
  train_df = pd.concat(train_dfs, ignore_index=True)
  eval_df = pd.concat(eval_dfs, ignore_index=True)
  # massaging configs to store all experiment parameters.
  config_df = pd.json_normalize(config.to_dict(), sep='_')
  complete_eval_df = data_utils.merge_pd_tiled_config(eval_df, config_df)
  complete_train_df = data_utils.merge_pd_tiled_config(train_df, config_df)
  if config.results.save_results:
    results_dir = config.results.experiment_dir.replace(
        '%CURRENT_DATE', current_date
    )
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    results_filename = config.results.filename.replace(
        '%JOB_ID_%TASK_ID', str(config.job_id)+'_'+str(config.task_id)
    )
    save_path = os.path.join(results_dir, results_filename)
    complete_train_df.to_csv(save_path + '_train.csv')
    complete_eval_df.to_csv(save_path + '_eval.csv')
    for sequence_name, final_mps in mps_sequences.items():
      mps_ds = data_utils.split_complex_ds(mps_utils.mps_to_xarray(final_mps))
      mps_ds.to_netcdf(save_path + f'_mps_{sequence_name}.nc')
  return complete_train_df, complete_eval_df, mps_sequences


def main(argv):
  config = FLAGS.train_config
  return run_full_batch_experiment(config)


if __name__ == '__main__':
  app.run(main)
