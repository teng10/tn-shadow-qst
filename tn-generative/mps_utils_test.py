"""Tests for mps_utils."""
import numpy as np
import quimb as qu
import quimb.tensor as qtn

import mps_utils

from absl.testing import absltest

HADAMARD = qu.gen.operators.hadamard()
Y_HADAMARD = np.roll(HADAMARD, 1, 0) * np.array([[1.0, 1.0j]])
EYE = qu.gen.operators.eye(2)

class MpoUtilsTests(absltest.TestCase):
  def test_z_to_basis_mpo(self, size):
    random_mps = qtn.MPS_rand_state(size, bond_dim=5)
    random_basis = np.random.randint(0, 2, size)
    mpo = mps_utils.z_to_basis_mpo(random_basis)
    mps_new_basis = mpo.apply(random_mps)
    mps_new_vector = mps_new_basis.to_dense()
    # build ED vector
    random_vector = random_mps.to_dense()
    rotation_matrices = [[HADAMARD, Y_HADAMARD, EYE][i] for i in random_basis]
    rotation_matrix = rotation_matrices[0]
    for mat in rotation_matrices[1:]:
      rotation_matrix = np.kron(rotation_matrix, mat)
    vector_new_basis = np.dot(rotation_matrix, random_vector)
    np.testing.assert_allclose(mps_new_vector/mps_new_vector[0], vector_new_basis/vector_new_basis[0])

  if __name__ == '__main__':
    absltest.main()