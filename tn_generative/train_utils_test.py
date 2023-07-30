"""Integration tests for train_utils.py."""
from absl.testing import absltest

from jax import config as jax_config
import quimb.tensor as qtn

from tn_generative.train_configs  import surface_code_training_config
from tn_generative import run_training


class RunTrainingTests(absltest.TestCase):
  """Tests data generation."""

  def setUp(self): 
    jax_config.update('jax_enable_x64', True)
    qtn.contraction.contract_backend('jax')  # set backend for current thread 
    self.experiment_config = surface_code_training_config.get_config()
    self.experiment_config.output.save_data = False


  def test_full_batch_experiment(self):
    run_training.run_full_batch_experiment(self.experiment_config)


  if __name__ == '__main__':
    absltest.main()
