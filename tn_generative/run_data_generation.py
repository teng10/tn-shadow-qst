"""Main file for running data generation."""
from absl import app
from absl import flags
import functools
from datetime import datetime
import os

from ml_collections import config_flags
import numpy as np
import jax
import haiku as hk
import quimb.tensor as qtn
import xarray as xr
import xyzpy

from tn_generative import data_generation
from tn_generative import data_utils
from tn_generative import mps_sampling
from tn_generative import mps_utils
from tn_generative import types

config_flags.DEFINE_config_file('data_config')
FLAGS = flags.FLAGS

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY


def generate_data(config):
  config.update_from_flattened_dict(config.sweep_param[config.task_id])
  current_date = datetime.now().strftime('%m%d')
  dtype = DTYPES_REGISTRY[config.dtype]
  task_system = TASK_REGISTRY[config.task.name](**config.task.kwargs)
  task_mpo = task_system.get_ham()
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
  # ds = ds.assign_attrs(**config)  #Can't save nested dictionary to netcdf.
  # TODO(YT): figure out how to unflatten config using pd.json.normalize.
  ds = ds.assign_attrs(**config.task.kwargs)
  ds.attrs['name'] = config.task.name
  # Saving data  
  if config.output.save_data:
    data_dir = config.output.data_dir.replace('%CURRENT_DATE', current_date)
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)    
    ds = data_utils.split_complex_ds(ds)
    # filename = os.path.join(
    #     config.output.data_dir,
    #     config.output.filename % config.task_id + '.nc'
    # )  #TODO: remove
    filename = os.path.join(data_dir, config.output.filename)
    filename = filename.replace('%JOB_ID', str(config.job_id))
    ds.to_netcdf(filename + '.nc')
  return ds


def main(argv):
  config = FLAGS.data_config
  return generate_data(config)


if __name__ == '__main__':
  app.run(main)
  