"""Tests for data_utils.py."""
import json
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import xarray as xr

from tn_generative import data_utils
from tn_generative import physical_systems


class PhysicalSystemToAttrsTest(parameterized.TestCase):
  """Tests for conversion between physical system and its attributes."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(3, 3)
      },
      {
          'testcase_name': 'ruby_pxp',
          'physical_system': physical_systems.RubyRydbergPXP(
              Lx=2, Ly=2, boundary='periodic', delta=np.ones(()) * 5),
      },
  )
  def test_roundtrip_to_attrs_and_back(self, physical_system):
    """Test a roundtrip and serialization of physical system."""
    with self.subTest('to_attrs_dict'):
      attrs_dict = data_utils.physical_system_to_attrs_dict(physical_system)
    with self.subTest('test_serializable'):
      json.dumps(attrs_dict)
    with self.subTest('reconstruction'):
      reconstructed = data_utils.physical_system_from_attrs_dict(attrs_dict)
      roundtrip_dict = data_utils.physical_system_to_attrs_dict(reconstructed)
      self.assertDictEqual(roundtrip_dict, attrs_dict)


class DatasetComputationTest(absltest.TestCase):
  """Tests for performing computations on xarray datasets."""

  def test_stream_mean_over_dim(self):
    """Test streaming mean computation over a dimension."""

    with self.subTest('single variable'):
      ds = xr.Dataset({
          'var1': xr.DataArray(np.arange(10), dims=['x']),
      })
      mean = data_utils.stream_mean_over_dim(ds, lambda x: x.var1, 'x')
      np.testing.assert_allclose(mean.values, np.mean(ds.var1.values))

    with self.subTest('computation with two variables and dims'):
      np.random.seed(41)
      ds = xr.Dataset(
          {
              'var1': (['x', 'y'], np.random.randint(0, 2, size=(200, 3))),
              'var2': (['x', 'y'], np.random.randint(0, 3, size=(200, 3))),
          },
          coords={'x': np.arange(200), 'y': np.arange(3)}
      )
      mean = data_utils.stream_mean_over_dim(
          ds, lambda x: x.var1 * x.var2, 'x', batch_size=25
      ).values
      expected = (ds.var1 * ds.var2).mean(dim='x').values
      np.testing.assert_allclose(mean, expected, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
