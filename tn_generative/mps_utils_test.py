"""Tests for mps_utils."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import quimb.tensor as qtn

from tn_generative  import mps_utils


class MpoUtilsTests(parameterized.TestCase):
  """Tests for matrix product operator utilities."""

  @parameterized.parameters(2, 3, 4, 6, 9)
  def test_z_to_basis_mpo(self, size):
    """Tests the rotation MPO by building an explicit vector rotation."""
    #TODO(YT): make random state deterministic.
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


  if __name__ == '__main__':
    absltest.main()