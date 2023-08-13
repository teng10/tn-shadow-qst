"""Integration tests for train_utils.py."""
from absl.testing import absltest
import os

from jax import config as jax_config
import quimb.tensor as qtn

from tn_generative.train_configs  import surface_code_training_config
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
    self.experiment_config = surface_code_training_config.get_config()
    self.experiment_config.output.save_data = False
    self.experiment_config.job_id = 1
    self.experiment_config.task_id = 0
    self.experiment_config.data.path = os.path.join(
        current_file_dir, 'test_data', 'example_dataset.nc'
    )
    self.experiment_config.data.num_training_samples = 1000
    self.experiment_config.training.num_training_steps = 10
    self.experiment_config.model.bond_dim = 5

  def test_full_batch_experiment(self):
    run_training.run_full_batch_experiment(self.experiment_config)


  if __name__ == '__main__':
    absltest.main()
