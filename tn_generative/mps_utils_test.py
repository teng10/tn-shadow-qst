"""Tests for mps_utils."""
from absl.testing import absltest
from absl.testing import parameterized
import functools
import os

import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen
import xarray as xr

from tn_generative import data_utils
from tn_generative  import mps_utils
from tn_generative import shadow_utils

class MpoUtilsTests(parameterized.TestCase):
  """Tests for matrix product operator utilities."""

  @parameterized.parameters(2, 3, 4, 6, 9)
  def test_z_to_basis_mpo(self, size, seed=42):
    """Tests the rotation MPO by building an explicit vector rotation.
    $\psi^\prime = U \psi$, where $U = \bigotimes_i U_i$ and $U_i$ is 
    the rotation matrix for the $i$-th site.
    """
    qugen.rand.seed_rand(seed)
    random_mps = qtn.MPS_rand_state(size, bond_dim=5)
    random_basis = np.random.randint(0, 2, size)
    mpo = mps_utils.z_to_basis_mpo(random_basis)
    rotated_mps = mpo.apply(random_mps)
    actual_rotated_mps_vector = rotated_mps.to_dense()
    # build ED vector
    random_vector = random_mps.to_dense()
    rotation_options = [mps_utils.HADAMARD, mps_utils.Y_HADAMARD, mps_utils.EYE]
    rotation_matrices = [rotation_options[i] for i in random_basis]
    rotation_matrix = functools.reduce(np.kron, rotation_matrices)
    expected_rotated_vector = np.dot(rotation_matrix, random_vector)
    np.testing.assert_allclose(
        actual_rotated_mps_vector, expected_rotated_vector, atol=1e-6)

  @parameterized.parameters(2, 3, 4, 5)
  def test_amplitude_via_contraction(self, size):
    """Tests the amplitude of an MPS state for explicit states."""
    with self.subTest("ghz_state"):
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

    with self.subTest("neel_state"):
      neel_state = qtn.tensor_builder.MPS_neel_state(size, down_first=True)
      measurement = (1. + (-1.)**np.arange(size)) / 2.
      actual_amplitude = mps_utils.amplitude_via_contraction(
          neel_state, measurement)
      np.testing.assert_allclose(actual_amplitude, 1.0)

  @parameterized.parameters(2, 3, 4, 5)
  def test_amplitude_via_contraction_basis(self, size):
    """Tests the amplitude of an MPS state in different basis."""
    with self.subTest("product_state"):  #TODO (YT): add random state tests.
      product_state = qtn.tensor_builder.MPS_computational_state(
          np.zeros(size, dtype=int))
      rng = np.random.default_rng()
      measurement = rng.integers(2, size=size)
      actual_amplitude = mps_utils.amplitude_via_contraction(
          product_state, measurement, basis=np.zeros(size))
      expected_amplitude = 1.0 / np.sqrt(2**size)
      np.testing.assert_allclose(actual_amplitude, expected_amplitude,
                                 atol=1e-6, rtol=1e-6)


class MpsUtilsTests(parameterized.TestCase):
  """Tests utils for matrix product states."""

  @parameterized.parameters(2, 3, 4, 5)
  def test_normalization(self, size):
    """Test normalization of MPS by comparing with quimb default."""
    np.random.seed(42)
    qugen.rand.seed_rand(42)
    mps = qtn.MPS_rand_state(size, bond_dim=5)  # default is normalized.
    non_normalized_mps = qtn.MatrixProductState(
        arrays=[np.random.randn(*x.shape) for x in mps.arrays])
    normalized_mps = mps_utils.uniform_normalize(non_normalized_mps)
    np.testing.assert_allclose(
        normalized_mps.H @ normalized_mps, 1.0, atol=1e-6
    )

  @parameterized.parameters(3, 4, 5)
  def test_xr_mps_conversion(self, size):
    """Test conversion between xarray and quimb MPS."""
    np.random.seed(42)
    qugen.rand.seed_rand(42)
    mps = qtn.MPS_rand_state(size, bond_dim=5)
    mps_xr = mps_utils.mps_to_xarray(mps)
    mps_qu = mps_utils.xarray_to_mps(mps_xr)
    np.testing.assert_equal(mps.arrays, mps_qu.arrays)

  @parameterized.parameters(3, 4, 5)
  def test_mps_fix_gauge_same(self, size):
    """Test mps is the same after gauge fixing."""
    qugen.rand.seed_rand(42)
    mps = qtn.MPS_rand_state(size, bond_dim=5)
    fixed_arrays = mps_utils.fix_gauge_arrays(mps)
    fixed_mps = qtn.MatrixProductState(fixed_arrays)
    np.testing.assert_allclose(fixed_mps.H @ mps, 1.)

  @parameterized.parameters(3, 4, 5)
  def test_mps_tensors_round_trip(self, size):
    """Test conversion between mps and expanded tensors."""
    qugen.rand.seed_rand(42)
    mps = qtn.MPS_rand_state(size, bond_dim=5)
    tensors = mps_utils._mps_to_expanded_tensors(mps)
    mps_from_tesors = mps_utils._mps_from_extended_tensors(tensors)
    np.testing.assert_allclose(mps_from_tesors.H @ mps, 1.0)  


class ProjectSubsystemTests(absltest.TestCase):
  """Tests for projecting subsystem density matrices."""

  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_dir, 'test_data/bell_state_ds_x_or_z.nc')
    bell_state_ds = xr.load_dataset(ds_path)
    self.bell_state_ds = data_utils.combine_complex_ds(bell_state_ds)
    self.bell_state_mps = qtn.MPS_ghz_state(2)

  def test_project_subsystem_with_shadow(self):
    """Test projection of the density matrix onto a Pauli subspace."""
    subsystem = [0, 1]
    shadow_fn = shadow_utils._get_shadow_single_shot_fn(self.bell_state_ds)
    shadow_state_custom = shadow_utils.construct_subsystem_shadows(
        self.bell_state_ds, subsystem, shadow_fn
    )
    bell_state_projected_xz = mps_utils.construct_subsystem_operators(
        self.bell_state_mps, subsystem, 'XZI')
    np.testing.assert_allclose(
        bell_state_projected_xz, shadow_state_custom, atol=0.01
    )
    

  if __name__ == "__main__":
    absltest.main()
