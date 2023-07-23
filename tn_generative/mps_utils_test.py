"""Tests for mps_utils."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen

from tn_generative  import mps_utils


class MpoUtilsTests(parameterized.TestCase):
  """Tests for matrix product operator utilities."""

  @parameterized.parameters(2, 3, 4, 6, 9)
  def test_z_to_basis_mpo(self, size, seed=42):
    """Tests the rotation MPO by building an explicit vector rotation."""
    qugen.rand.seed_rand(seed)
    random_mps = qtn.MPS_rand_state(size, bond_dim=5) 
    random_basis = np.random.randint(0, 2, size)
    mpo = mps_utils.z_to_basis_mpo(random_basis)
    rotated_mps = mpo.apply(random_mps)
    actual_rotated_mps_vector = rotated_mps.to_dense()
    # build ED vector
    random_vector = random_mps.to_dense()
    rotation_options = [mps_utils.HADAMARD, mps_utils.Y_HADAMARD, mps_utils.EYE]
    rotation_matrices = [
        np.conjugate(rotation_options[i]).T
        for i in random_basis
    ]
    rotation_matrix = functools.reduce(np.kron, rotation_matrices)
    expected_rotated_vector = np.dot(rotation_matrix, random_vector)
    np.testing.assert_allclose(
        actual_rotated_mps_vector, expected_rotated_vector, atol=1e-6)

  @parameterized.parameters(2, 3, 4, 5)
  def test_amplitude_via_contraction(self, size):
    """Tests the amplitude of an MPS state for explicit states."""
    with self.subTest('ghz_state'):
      ghz_state = qtn.tensor_builder.MPS_ghz_state(size)
      measurement = np.zeros(size)
      actual_amplitude = mps_utils.amplitude_via_contraction(
          ghz_state, measurement)
      expected_amplitude = 1 / np.sqrt(2) 
      np.testing.assert_allclose(actual_amplitude, expected_amplitude)
      measurement = np.ones(size)
      actual_amplitude = mps_utils.amplitude_via_contraction(
          ghz_state, measurement)
      expected_amplitude = 1 / np.sqrt(2)
      np.testing.assert_allclose(actual_amplitude, expected_amplitude)
     
    with self.subTest('neel_state'):
      neel_state = qtn.tensor_builder.MPS_neel_state(size, down_first=True)
      measurement = (1. + (-1.)**np.arange(size)) / 2.
      actual_amplitude = mps_utils.amplitude_via_contraction(
          neel_state, measurement)
      np.testing.assert_allclose(actual_amplitude, 1.0)

  @parameterized.parameters(2, 3, 4, 5)
  def test_amplitude_via_contraction_basis(self, size):
    """Tests the amplitude of an MPS state in different basis."""
    with self.subTest('product_state'):  #TODO (YT): add random state tests.
      product_state = qtn.tensor_builder.MPS_computational_state(
          np.zeros(size, dtype=int))
      rng = np.random.default_rng()
      measurement = rng.integers(2, size=size)
      actual_amplitude = mps_utils.amplitude_via_contraction(
          product_state, measurement, basis=np.zeros(size))
      expected_amplitude = 1.0 / np.sqrt(2**size)
      np.testing.assert_allclose(actual_amplitude, expected_amplitude, 
                                 atol=1e-6, rtol=1e-6)
      

  if __name__ == "__main__":
    absltest.main()
