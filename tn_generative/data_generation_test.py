"""Integration tests for data_generation.py."""
import tempfile
from absl.testing import absltest

from ml_collections import config_dict

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
    """Set up config for data generation for testing."""
    self.config_dict = {}
    self.config_dict['job_id'] = 0
    self.config_dict['task_id'] = 0
    self.config_dict['task.kwargs'] = {
        'size_x': 3, 'size_y': 3, 'onsite_z_field': 0.1
    }
    self.config_dict['dmrg.bond_dims'] = 5
    self.config_dict['sampling.num_samples'] = 500
    self.config_dict['output.data_dir'] = tempfile.mkdtemp('temp')

  def test_generate_surface_code(self):
    """Tests data generation for surface code."""
    config = surface_code_data_config.get_config()
    config.update_from_flattened_dict(self.config_dict)
    run_data_generation.generate_data(config)

  def test_generate_ruby_pxp(self):
    """Tests data generation for rydberg hamiltonian PXP model."""
    config = ruby_pxp_data_config.get_config()
    config.update_from_flattened_dict(self.config_dict)
    run_data_generation.generate_data(config)

  #TODO(YT): finish cluster state PR
  # def test_generate_cluster_state(self):
  #   """Tests data generation for cluster state."""
  #   config = cluster_state_data_config.get_config()
  #   config.update_from_flattened_dict(self.config_dict)
  #   run_data_generation.generate_data(config)

  if __name__ == '__main__':
    absltest.main()
