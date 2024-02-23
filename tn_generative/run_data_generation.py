"""Main file for running data generation."""
  # run this script with the following command in the terminal:
  # python -m tn_generative.run_data_generation\
  # --data_config=tn_generative/data_configs/surface_code_data_config.py\
  # --data_config.job_id=0 --data_config.task_id=0
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

def _run_data_generation(
    init_seed: int,
    num_samples: int,
    sampling_method: str,
    mps: qtn.MatrixProductState,
) -> xr.Dataset:
  """Runs data generation for a given MPS.
  
  Args:
    init_seed: The initial seed for the random number generator.
    num_samples: The number of samples to generate.
    sampling_method: The method to use for sampling.
    mps: The MPS to sample from.
  
  Returns:
    An xarray dataset containing the generated samples.
  """
  qtn.contraction.set_tensor_linop_backend('jax')
  qtn.contraction.set_contract_backend('jax')
  rng_seq = hk.PRNGSequence(init_seed)
  all_keys = rng_seq.take(num_samples)
  sample_fn = mps_sampling.SAMPLER_REGISTRY[sampling_method]
  sample_fn = functools.partial(sample_fn, mps=mps)
  sample_fn = jax.jit(sample_fn, backend='cpu')
  generate_fn = lambda sample: sample_fn(all_keys[sample])
  runner = xyzpy.Runner(
      generate_fn,
      var_names=['measurement', 'basis'],
      var_dims={'measurement': ['site'], 'basis': ['site']},
      var_coords={'site': np.arange(mps.L)},
  )
  combos = {'sample': np.arange(num_samples)}
  return runner.run_combos(combos, parallel=False)

def generate_data(config):
  if config.sweep_name in config.sweep_fn_registry:
    sweep_param = config.sweep_fn_registry[config.sweep_name]
    config.update_from_flattened_dict(sweep_param[config.task_id])
  elif config.sweep_name is None:
    pass
  else:
    raise ValueError(f'{config.sweep_name} not in sweep_fn_registry.')
  current_date = datetime.now().strftime('%m%d')
  dtype = DTYPES_REGISTRY[config.dtype]
  task_system = TASK_REGISTRY[config.task.name](**config.task.kwargs)
  task_mpo = task_system.get_ham()
  if config.dmrg.run and (config.sampling.mps_filepath is None):
    # Running DMRG
    qtn.contraction.set_tensor_linop_backend('numpy')
    qtn.contraction.set_contract_backend('numpy')
    mps = qtn.MPS_rand_state(task_mpo.L, config.dmrg.bond_dims, dtype=dtype)
    dmrg = qtn.DMRG1(task_mpo, bond_dims=config.dmrg.bond_dims, p0=mps)
    convergence = dmrg.solve(**config.dmrg.solve_kwargs)
    mps = dmrg.state.copy()
    mps = mps.canonize(0)  # canonicalize MPS.
    # TODO(YT): add dmrg data analysis module.
    energy_variance = (
        mps.H @ (task_mpo.apply(task_mpo.apply(mps))) - dmrg.energy**2
    )
    mps_properties = data_utils.compute_onsite_pauli_expectations(mps, task_system)
    mps_properties['energy'] = dmrg.energy
    mps_properties['energy_variance'] = energy_variance
    mps_properties['entropy'] = mps.entropy(mps.L // 2)
    mps_properties['max_bond'] = mps.max_bond()
    mps_properties['convergence'] = int(convergence)
    mps_properties.attrs = data_utils.physical_system_to_attrs_dict(task_system)
    target_mps_ds = mps_utils.mps_to_xarray(mps)
    mps_properties = xr.merge([target_mps_ds, mps_properties],
        combine_attrs='no_conflicts'
    )
  elif (not config.dmrg.run) and (config.sampling.mps_filepath is not None):
    mps_ds = xr.load_dataset(config.sampling.mps_filepath)
    mps_ds = data_utils.combine_complex_ds(mps_ds)
    mps = mps_utils.xarray_to_mps(mps_ds)
    mps = mps.canonize(0)
    mps_properties = mps_ds.copy()
  else:
    raise ValueError(f'Either run DMRG or load MPS; but {config.dmrg.run=} and \
        loading mps is {config.sampling.mps_filepath=}'
    )
  # Running data generation
  ds = _run_data_generation(
      config.sampling.init_seed,
      config.sampling.num_samples,
      config.sampling.sampling_method,
      mps,
  )
  ds = xr.merge([ds, mps_properties], combine_attrs='no_conflicts')
  # Saving data
  if config.output.save_data:
    data_dir = config.output.data_dir.replace('%CURRENT_DATE', current_date)
    filename = config.output.filename.replace('%JOB_ID', str(config.job_id))
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    ds = data_utils.split_complex_ds(ds)
    filepath = os.path.join(data_dir, filename)
    #TODO(YT): remove/add extension e.g. os.path.splitext(name)[0] + '.nc'
    ds.to_netcdf(filepath + '.nc')
    # np.save(filepath, np.array(dmrg.energies))  #TODO(YT): add dmrg debug.
  return ds


def main(argv):
  config = FLAGS.data_config
  return generate_data(config)


if __name__ == '__main__':
  app.run(main)
