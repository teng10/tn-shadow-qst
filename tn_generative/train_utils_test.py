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
    self.experiment_config.job_id = 1
    self.experiment_config.task_id = 0
    self.experiment_config.data.path = self.experiment_config.data.path.replace(
        '%CURRENT_DATE', '0731'
    ).replace(
        '%JOB_ID_surface_code_%SYSTEM_SIZE_%D_%ONSITE_Z_FIELD', 
        '1_surface_code_system_size=3_d=10_onsite_z_field=0.000.nc'
    )
    self.experiment_config.data.num_training_samples = 1000
    self.experiment_config.training.num_training_steps = 10
    self.experiment_config.model.bond_dim = 5

  def test_full_batch_experiment(self):
    run_training.run_full_batch_experiment(self.experiment_config)


  if __name__ == '__main__':
    absltest.main()
