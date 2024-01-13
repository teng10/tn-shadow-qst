"""Integration tests for train_utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import os

from jax import config as jax_config
import quimb.tensor as qtn

from tn_generative.train_configs  import surface_code_train_config
from tn_generative.train_configs import ruby_pxp_train_config
from tn_generative import run_training


class RunTrainingTests(parameterized.TestCase):
  """Tests training pipeline. """
  #TODO(YT): currently sweep is tested only at the level of config file.
  # TODO (YT): consider separatly testing the training schemes. 
  # Default `run_training`` uses a schedule with 
  # 1) minibatch sgd (regularization free) and
  # 2) full batch optimization.

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
        'data.num_training_samples': 100,
        'data.num_test_samples': 200,
        'training.steps_sequence': (5, 2),
        'model.bond_dim': 2,
    }

  def test_surface_code(self):
    """Tests training for surface code using default training sequence."""
    config = surface_code_train_config.get_config()
    config.update_from_flattened_dict(self.default_options)
    config.data.filename = 'example_dataset.nc'
    run_training.run_full_batch_experiment(config)

  def test_ruby_pxp(self):
    """Tests training for ruby PXP model  using default training sequence."""
    config = ruby_pxp_train_config.get_config()
    config.update_from_flattened_dict(self.default_options)
    config.data.filename = 'ruby_pxp_data.nc'
    run_training.run_full_batch_experiment(config)

  # def test_ruby_pxp_density_regularization(self):
  #   """Tests training for ruby PXP model with density matirx regularization."""
  #   config = ruby_pxp_train_config.get_config()
  #   config.update_from_flattened_dict(self.default_options)
  #   config.data.filename = 'ruby_pxp_data.nc'
  #   reg_name = 'reduced_density_matrices'
  #   config.training.training_schemes.lbfgs_reg.reg_name = reg_name
  #   config.training.training_schemes.lbfgs_reg.reg_kwargs = {
  #       'beta': 1., 'estimator': 'mps'
  #   }
  #   run_training.run_full_batch_experiment(config)

  if __name__ == '__main__':
    absltest.main()
