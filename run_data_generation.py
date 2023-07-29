"""Main file for running data generation."""
from absl import app
from absl import flags
import functools
from datetime import datetime

from ml_collections import config_flags
import numpy as np
import jax
import haiku as hk
import quimb.tensor as qtn
import xarray as xr
import xyzpy

from tn_generative import data_generation
from tn_generative import mps_sampling
from tn_generative import mps_utils
from tn_generative import types

config_flags.DEFINE_config_file('config')
FLAGS = flags.FLAGS

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY


def main(argv):
  current_date = datetime.now().strftime('%m%d')
  config = FLAGS.config
  dtype = DTYPES_REGISTRY[config.dtype]
  task_system = TASK_REGISTRY[config.task.name](**config.task.kwargs)
  task_mpo = task_system.get_ham_mpo()
  # Running DMRG
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
  ds = xr.merge([target_mps_ds, ds])
  # Saving data  #TODO(YT): add utils for saving `complex` type data
  # ds.to_netcdf(config.output.data_save_path + 
  #     f'{current_date}_{config.output.filename}' + '.nc'
  # )


if __name__ == '__main__':
  app.run(main)
  