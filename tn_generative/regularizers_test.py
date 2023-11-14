"""Tests for data_utils.py."""
import os
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import xarray as xr

from tn_generative import data_utils
from tn_generative import physical_systems
from tn_generative import regularizers


class SubsystemsRegularizerTest(parameterized.TestCase):
  """Tests for regularization using subsystems."""
  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/example_dataset.nc')
    test_ds = xr.load_dataset(ds_path)
    self.test_ds = data_utils.combine_complex_ds(test_ds)
    self.reg_fn = regularizers.REGULARIZER_REGISTRY['subsystems']

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(4, 2)
      },
      # {
      #     'testcase_name': 'ruby_pxp',
      #     'physical_system': physical_systems.RubyRydbergPXP(
      #         Lx=2, Ly=np.int32(5), boundary='periodic', delta=np.ones(()) * 5),
      # },
  )
  def test_surface_code_explicit(self, physical_system):
    """"""
    subsystems = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]]
    subsystem_kwargs = {'method': 'explicit', 'explicit_subsystems': subsystems}
    self.reg_fn(
        physical_system, self.test_ds,
        subsystem_kwargs=subsystem_kwargs
    )



if __name__ == '__main__':
  absltest.main()