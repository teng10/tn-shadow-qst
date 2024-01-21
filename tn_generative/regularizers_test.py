"""Tests for regularizers.py."""
import os
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import xarray as xr
import quimb.tensor as qtn

from tn_generative import data_utils
from tn_generative import mps_utils
from tn_generative import physical_systems
from tn_generative import regularizers


class HamiltonianRegularizerTest(parameterized.TestCase):
  """Tests for regularization using reduced_density_matrices."""
  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/example_dataset.nc')
    test_ds = xr.load_dataset(ds_path)
    self.test_ds = data_utils.combine_complex_ds(test_ds)
    self.get_reg_fn = regularizers.REGULARIZER_REGISTRY[
        'hamiltonian'
    ]

  @parameterized.product(
      ({
          'physical_system': physical_systems.SurfaceCode(3, 3)
      }, ), 
      ({'method': 'mps', 'atol': 1e-6}, {'method': 'shadow', 'atol': 1e-3})
  )
  def test_hamiltonian_regularization_explicit(
      self, physical_system, method, atol
  ):
    """Tests that regularization is positive and zero for true state."""
    with self.subTest(f'{method=}'):
      reg_fn = self.get_reg_fn(physical_system, self.test_ds, method=method)
      with self.subTest('assert_positive'):
        mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
        reg_val = reg_fn(mps_rand.arrays)
        self.assertGreater(reg_val, 0.)

      with self.subTest('assert_zero'):
        mps = mps_utils.xarray_to_mps(self.test_ds).arrays
        np.testing.assert_allclose(reg_fn(mps), 0., atol=atol)    


class SubsystemsRegularizerTest(parameterized.TestCase):
  """Tests for regularization using reduced_density_matrices."""
  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/example_dataset.nc')
    test_ds = xr.load_dataset(ds_path)
    self.test_ds = data_utils.combine_complex_ds(test_ds)
    self.get_reg_fn = regularizers.REGULARIZER_REGISTRY[
        'reduced_density_matrices'
    ]

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(3, 3)
      }
  )
  def test_reduced_density_matrices_explicit(self, physical_system):
    """Tests that regularization is positive and zero for true state."""
    subsystems = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]]
    subsystem_kwargs = {'method': 'explicit', 'explicit_subsystems': subsystems}
    reg_fn = self.get_reg_fn(physical_system, self.test_ds,
        subsystem_kwargs=subsystem_kwargs
    )
    with self.subTest('assert_positive'):
      mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
      reg_val = reg_fn(mps_rand.arrays)
      # Consider remove this test as this depends on the random seed.
      np.testing.assert_allclose(reg_val, 0.835518269281676, atol=1e-6)

    with self.subTest('assert_zero'):
      mps = mps_utils.xarray_to_mps(self.test_ds).arrays
      np.testing.assert_allclose(reg_fn(mps), 0., atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ruby_pxp',
          'physical_system': physical_systems.RubyRydbergPXP(2, 2)
      }
  )
  def test_reduced_density_matrices_default(self, physical_system):
    """."""
    subsystem_kwargs = {'method': 'default'}
    mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
    ds = mps_utils.mps_to_xarray(mps_rand)
    reg_fn = self.get_reg_fn(physical_system, ds,
        subsystem_kwargs=subsystem_kwargs
    )
    with self.subTest('assert_positive'):
      mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=43)
      reg_val = reg_fn(mps_rand.arrays)
      self.assertGreater(reg_val, 0.)


if __name__ == '__main__':
  absltest.main()
