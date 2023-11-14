"""Integration tests for train_utils.py."""
from absl.testing import absltest
import os

from jax import config as jax_config
import quimb.tensor as qtn

from tn_generative.train_configs  import surface_code_train_config
from tn_generative.train_configs import ruby_pxp_train_config
from tn_generative import run_training


class RunTrainingTests(absltest.TestCase):
  """Tests data generation."""

  def setUp(self): 
    jax_config.update('jax_enable_x64', True)
    qtn.contraction.contract_backend('jax')  # set backend for current thread 
    # Get the path of the currently executing script
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)    
    
    self.default_options = {
        'results.save_results': False,
        'job_id': 0,
        'task_id': 0,
        'data.dir': os.path.join(current_file_dir, 'test_data'),
        'data.filename': 'example_dataset.nc',
        'data.num_training_samples': 1000,
        'training.num_training_steps': 10,
        'model.bond_dim': 5,
    }

  def test_surface_code(self):
    """Tests training for surface code."""
    config = surface_code_train_config.get_config()
    config.update_from_flattened_dict(self.default_options)
    run_training.run_full_batch_experiment(config)

  def test_ruby_pxp(self):
    """Tests training for ruby PXP model."""
    config = ruby_pxp_train_config.get_config()
    config.update_from_flattened_dict(self.default_options)
    run_training.run_full_batch_experiment(config)


  if __name__ == '__main__':
    absltest.main()
