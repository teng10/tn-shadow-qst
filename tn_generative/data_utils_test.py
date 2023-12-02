"""Tests for data_utils.py."""
import json
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from tn_generative import data_utils
from tn_generative import physical_systems


class PhysicalSystemToAttrsTest(parameterized.TestCase):
  """Tests for conversion between physical system and its attributes."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(1, 2)
      },
      {
          'testcase_name': 'ruby_pxp',
          'physical_system': physical_systems.RubyRydbergPXP(
              Lx=2, Ly=np.int32(5), boundary='periodic', delta=np.ones(()) * 5),
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


if __name__ == '__main__':
  absltest.main()
