"""Integration tests for data_generation.py."""
import tempfile
from absl.testing import absltest

from tn_generative.data_configs import surface_code_data_config
from tn_generative.data_configs import ruby_pxp_data_config
from tn_generative  import data_generation
from tn_generative  import run_data_generation
from tn_generative  import types

DTYPES_REGISTRY = types.DTYPES_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY


class RunDataGeneration(absltest.TestCase):
  """Tests data generation."""

  def setUp(self):
    """Set up config for data generation."""
    self.default_options = {
        'job_id': 0,
        'task_id': 0,
        'dmrg.bond_dims': 5,
        'sampling.num_samples': 500,
        'output.data_dir': tempfile.mkdtemp('temp'),
    }

  def test_generate_surface_code(self):
    """Tests data generation for surface code."""
    config = surface_code_data_config.get_config()
    config.task.kwargs = {'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.1}
    config.update_from_flattened_dict(self.default_options)
    run_data_generation.generate_data(config)

  def test_ruby_pxp(self):
    """Tests data generation for ruby lattice PXP model."""
    config = ruby_pxp_data_config.get_config()
    config.task.kwargs = {'size_x': 1, 'size_y': 2, 'delta': 0.1}
    config.update_from_flattened_dict(self.default_options)
    run_data_generation.generate_data(config)

  if __name__ == '__main__':
    absltest.main()
