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
    self.get_reg_fn = regularizers.REGULARIZER_REGISTRY['reduced_density_matrices']

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
      np.testing.assert_allclose(reg_val, 0.835518269281676)

    with self.subTest('assert_zero'):
      mps = mps_utils.xarray_to_mps(self.test_ds).arrays
      self.assertEqual(reg_fn(mps), 0.)


if __name__ == '__main__':
  absltest.main()