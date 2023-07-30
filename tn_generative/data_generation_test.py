"""Integration tests for data_generation.py."""
import functools
from absl.testing import absltest

from tn_generative.data_configs import surface_code_data_config
from tn_generative  import data_generation
from tn_generative  import run_data_generation
from tn_generative  import types

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY


class RunDataGeneration(absltest.TestCase):
  """Tests data generation."""

  def setUp(self):
    """Set up config for data generation using surface code."""
    self.config = surface_code_data_config.get_config()
    self.config.output.save_data = False

  def test_generate_surface_code(self):
    """Tests data generation for surface code."""
    config = self.config  #TODO(YT): move to run_data_generation.py
    run_data_generation.generate_data(config)


  if __name__ == '__main__':
    absltest.main()
