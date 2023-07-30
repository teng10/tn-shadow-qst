"""Integration tests for train_utils.py."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from jax import config as jax_config
import quimb.tensor as qtn
import quimb.gen as qugen
import pandas as pd

from tn_generative.train_configs  import surface_code_training_config
from tn_generative.data_configs  import surface_code_data_config
from tn_generative  import mps_utils
from tn_generative  import data_generation
from tn_generative  import train_utils
from tn_generative import regularizers
from tn_generative  import types
from tn_generative import run_data_generation

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY
REGULARIZER_REGISTRY = regularizers.REGULARIZER_REGISTRY


class RunTrainingTests(parameterized.TestCase):
  """Tests data generation."""

  def setUp(self):
    # Generate data for training.  #TODO(YT): use config file.
    jax_config.update('jax_enable_x64', True)
    config = surface_code_data_config.get_config()
    config.output.save_data = False
    self.ds = run_data_generation.generate_data(config)

  def run_full_batch_experiment(self, exp_config, ds):
    #TODO(YT): move to run_train.py
    train_config = exp_config.training
    model_config = exp_config.model
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

  def test_full_batch_experiment(self):
    experiment_config = surface_code_training_config.get_config()
    qtn.contraction.contract_backend('jax')  # set backend for current thread
    self.run_full_batch_experiment(experiment_config, self.ds)


  if __name__ == '__main__':
    absltest.main()
