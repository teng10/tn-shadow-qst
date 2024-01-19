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


class SubsystemsRegularizerTest(parameterized.TestCase):
  """Tests for regularization using reduced_density_matrices."""
  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/example_dataset.nc')
    test_ds = xr.load_dataset(ds_path)
    self.test_ds = data_utils.combine_complex_ds(test_ds)
    ruby_path = os.path.join(current_file_dir, 'test_data/ruby_pxp_data.nc')
    ds_ruby = xr.load_dataset(ruby_path)
    self.ds_ruby = data_utils.combine_complex_ds(ds_ruby)
    #TODO(YT): make it parameterized. Right now just swapping the tests...
    # self.get_reg_fn = regularizers.REGULARIZER_REGISTRY[
    #     'reduced_density_matrices'
    # ]
    self.get_reg_fn = regularizers.REGULARIZER_REGISTRY[
        'subsystem_xz_operators'
    ]

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(3, 3)
      }
  )
  def test_projected_reduced_density_matrices(self, physical_system):
    """Tests that regularization is positive and zero for true state."""
    subsystems = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]]
    subsystem_kwargs = {'method': 'explicit', 'explicit_subsystems': subsystems}
    reg_fn = self.get_reg_fn(physical_system, self.test_ds,
        subsystem_kwargs=subsystem_kwargs, method='mps',
    )
    with self.subTest('assert_positive'):
      mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
      reg_val = reg_fn(mps_rand.arrays)
      self.assertGreater(reg_val, 0.)

    with self.subTest('assert_zero'):
      mps = mps_utils.xarray_to_mps(self.test_ds).arrays
      np.testing.assert_allclose(reg_fn(mps), 0., atol=1e-6)

    with self.subTest('shadow method assert positive'):
      reg_fn = self.get_reg_fn(physical_system, self.test_ds,
          subsystem_kwargs=subsystem_kwargs, method='shadow',
      )
      mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
      reg_val = reg_fn(mps_rand.arrays)
      self.assertGreater(reg_val, 0.)



  # TODO: fix this test. Current dataset is fixed XZ. Need random XZ.
  # @parameterized.named_parameters(
  #     {
  #         'testcase_name': 'ruby_pxp',
  #         'physical_system': physical_systems.RubyRydbergPXP(2, 1)
  #     }
  # )
  # def test_reduced_density_matrices_default(self, physical_system):
  #   """."""
  #   subsystem_kwargs = {'method': 'default'}
  #   mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=42)
  #   # ds = mps_utils.mps_to_xarray(mps_rand)
  #   reg_fn = self.get_reg_fn(physical_system, self.ds_ruby,
  #       subsystem_kwargs=subsystem_kwargs
  #   )
  #   with self.subTest('assert_positive'):
  #     mps_rand = qtn.MPS_rand_state(physical_system.n_sites, 2, seed=43)
  #     reg_val = reg_fn(mps_rand.arrays)
  #     self.assertGreater(reg_val, 0.)


if __name__ == '__main__':
  absltest.main()
