"""Integration tests for train_utils.py."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
from jax import config as jax_config
import haiku as hk
import quimb.tensor as qtn
import quimb.gen as qugen
import xarray as xr
import pandas as pd
import xyzpy
from ml_collections import config_dict

from tn_generative  import mps_utils
from tn_generative  import mps_sampling
from tn_generative  import data_generation
from tn_generative  import train_utils
from tn_generative import regularizers
from tn_generative  import typing

DTYPES_REGISTRY = typing.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY
REGULARIZER_REGISTRY = regularizers.REGULARIZER_REGISTRY


class RunDataGeneration(parameterized.TestCase):
  """Tests data generation."""

  def setUp(self):
    # Generate data for training.  #TODO(YT): use config file.
    jax_config.update('jax_enable_x64', True)
    def surface_code_config():
      config = config_dict.ConfigDict()
      # Task configuration.
      config.dtype = 'complex128'
      config.task = config_dict.ConfigDict()
      config.task.name = 'surface_code'
      config.task.kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.1}
      # DMRG configuration.
      config.dmrg = config_dict.ConfigDict()
      config.dmrg.bond_dims = 5
      config.dmrg.solve_kwargs = {
          'max_sweeps': 40, 'cutoffs': 1e-6, 'verbosity': 1
      }
      # Sampler configuration.
      config.sampling = config_dict.ConfigDict()
      config.sampling.sampling_method = 'xz_basis_sampler'
      config.sampling.init_seed = 42
      config.sampling.num_samples = 500
      return config

    config = surface_code_config()  #TODO(YT): move to run_data_generation.py
    dtype = DTYPES_REGISTRY[config.dtype]
    task_system = TASK_REGISTRY[config.task.name](**config.task.kwargs)
    task_mpo = task_system.get_ham_mpo()
    qtn.contraction.set_tensor_linop_backend('numpy')
    qtn.contraction.set_contract_backend('numpy')
    mps = qtn.MPS_rand_state(task_mpo.L, config.dmrg.bond_dims, dtype=dtype)
    dmrg = qtn.DMRG1(task_mpo, bond_dims=config.dmrg.bond_dims, p0=mps)
    dmrg.solve(**config.dmrg.solve_kwargs)
    mps = dmrg.state.copy()
    mps = mps.canonize(0)  # canonicalize MPS.

    # Running data generation
    qtn.contraction.set_tensor_linop_backend('jax')
    qtn.contraction.set_contract_backend('jax')

    rng_seq = hk.PRNGSequence(config.sampling.init_seed)
    all_keys = rng_seq.take(config.sampling.num_samples)

    sample_fn = mps_sampling.SAMPLER_REGISTRY[config.sampling.sampling_method]
    sample_fn = functools.partial(sample_fn, mps=mps)
    sample_fn = jax.jit(sample_fn, backend='cpu')
    generate_fn = lambda sample: sample_fn(all_keys[sample])

    runner = xyzpy.Runner(
        generate_fn,
        var_names=['measurement', 'basis'],
        var_dims={'measurement': ['site'], 'basis': ['site']},
        var_coords={'site': np.arange(mps.L)},
    )
    combos = {
        'sample': np.arange(config.sampling.num_samples),
    }
    ds = runner.run_combos(combos, parallel=False)

    target_mps_ds = mps_utils.mps_to_xarray(mps)
    self.ds = xr.merge([target_mps_ds, ds])

  def get_experiment_config(self):    #TODO(YT): move to config_training.py
    config = config_dict.ConfigDict()
    config.model = config_dict.ConfigDict()
    config.model.bond_dim = 5
    config.model.dtype = 'complex128'
    config.model.init_seed = 43
    # data.
    config.data = config_dict.ConfigDict()
    config.data.num_training_samples = 1000
    # training.
    config.training = config_dict.ConfigDict()
    config.training.num_training_steps = 10
    config.training.opt_kwargs = {}
    config.training.reg_name = 'surface_code'
    config.training.reg_kwargs = {'beta': 1.}
    config.training.reg_strength = 1.
    config.training.estimator = 'mps'
    # physical system.
    config.task_name = 'surface_code'
    # use zero field surface code to get only stabilizers.
    config.task_kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.}
    return config


  def run_full_batch_experiment(self, exp_config, ds):
    #TODO(YT): move to run_train.py
    train_config = exp_config.training
    model_config = exp_config.model
    train_ds = ds.isel(sample=slice(0, exp_config.data.num_training_samples))
    system = TASK_REGISTRY[exp_config.task_name](**exp_config.task_kwargs)
    estimator_fn = functools.partial(
        mps_utils.estimate_observable, method=train_config.estimator
    )
    reg_fn = REGULARIZER_REGISTRY[train_config.reg_name]
    reg_fn = reg_fn(system=system,
        estimator_fn=estimator_fn, train_ds=train_ds, **train_config.reg_kwargs
    )
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
    experiment_config = self.get_experiment_config()
    qtn.contraction.contract_backend('jax')  # set backend for current thread
    self.run_full_batch_experiment(experiment_config, self.ds)


  if __name__ == '__main__':
    absltest.main()
