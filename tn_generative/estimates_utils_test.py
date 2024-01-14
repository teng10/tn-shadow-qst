"""Tests for estimates_utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import os

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import quimb.gen as qugen
import xarray as xr

from tn_generative import data_utils
from tn_generative import estimates_utils
from tn_generative import physical_systems


class ExtractNonIdentityMPO(parameterized.TestCase):
  """Tests for extraction of non-identity part of MPO."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'surface_code',
          'physical_system': physical_systems.SurfaceCode(3, 3)
      },
      {
          'testcase_name': 'ruby_pxp',
          'physical_system': physical_systems.RubyRydbergPXP(
              Lx=2, Ly=2, boundary='periodic', delta=np.ones(()) * 5),
      },
  )
  def test_subsystem_expectation_vals(self, physical_system,
      seed=42, bond_dim=2
  ):
    """Test expectation values of subsystems of random MPS."""
    mpos = physical_system.get_ham_mpos()
    qugen.rand.seed_rand(seed)
    random_mps = qtn.MPS_rand_state(physical_system.n_sites, bond_dim)
    for mpo in mpos:
      mps = random_mps.copy()
      expected_ev = (mps.H @ (mpo.apply(mps)))
      subsystem_mpo, sub_indices = estimates_utils._extract_non_identity_mpo(
          mpo, return_indices=True
      )
      if sub_indices == []: # skip identity mpo
        continue
      else:
        sub_rdm = (mps.partial_trace(sub_indices)).to_dense()
        actual_ev = np.trace(sub_rdm @ subsystem_mpo.to_dense())
        np.testing.assert_allclose(expected_ev, actual_ev)


  @parameterized.parameters(3, 4, 5, 6)
  def test_operator_dense(self, L):
    """Test extraction of non-identity part of MPO."""
    mpo = qtn.MPO_product_operator([
        qu.pauli('I')] * (L - 3) + [qu.pauli('X'), qu.pauli('I'), qu.pauli('Y')
    ])
    subsystem_mpo, _ = estimates_utils._extract_non_identity_mpo(
        mpo, return_indices=True
    )
    expected_op = np.kron(qu.pauli('x'), qu.pauli('y'))
    np.testing.assert_allclose(expected_op, subsystem_mpo.to_dense())


class EstimateExpvalPauliFromMeasurements(parameterized.TestCase):
  """Tests for estimating expectation values of pauli words."""

  def setUp(self):
    self.num_samples = 1000
    # Load bell state dataaset.
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/bell_state_ds.nc')
    self.bell_state_ds = xr.load_dataset(ds_path)
    # Load surface code dataset.
    ds_path = os.path.join(current_file_dir, 'test_data/surface_code_xz.nc')
    surface_code_ds = xr.load_dataset(ds_path)
    surface_code_ds = surface_code_ds.isel(sample=slice(0, self.num_samples))
    self.surface_code_ds = data_utils.combine_complex_ds(surface_code_ds)

  def test_product_state(self):
    """Test expectation values of pauli words on product state."""
    with self.subTest('neel'):
      # Neel state. Expectation value of Z is -1.
      pauli = np.array([2, 2, 2])
      sub_indices = [0, 1, 2]
      measurements_z = np.tile(np.array([0, 1, 0]), (self.num_samples, 1))
      ds = xr.Dataset({
          'measurement': (['sample', 'site'], measurements_z),
          'basis': (['sample', 'site'], 2. * np.ones((self.num_samples, 3))),
      })
      expected_ev = -1.
      actual_ev = estimates_utils.estimate_expval_pauli_from_measurements(
          ds, pauli, sub_indices, estimator='empirical'
      )
      np.testing.assert_allclose(expected_ev, actual_ev)

    with self.subTest('000'):
      # Computational product state with expectation value of Z is 1.
      pauli = np.array([2, 2, 2])
      sub_indices = [0, 1, 2]
      measurements_z = np.tile(np.array([0, 0, 0]), (self.num_samples, 1))
      ds = xr.Dataset({
          'measurement': (['sample', 'site'], measurements_z),
          'basis': (['sample', 'site'], 2. * np.ones((self.num_samples, 3))),
      })
      expected_ev = 1.
      actual_ev = estimates_utils.estimate_expval_pauli_from_measurements(
          ds, pauli, sub_indices, estimator='empirical'
      )

  def test_bell_state(self):
    """Test expectation values of pauli words on Bell state."""
    # Test expectation value of stabilizers on bell state of size 2.
    if self.bell_state_ds.sizes['site'] != 2:
      raise ValueError(f"Bell state dataset is \
          {self.bell_state_ds.sizes['site']} not of size 2."
      )
    with self.subTest('pauli z'):
      # Expectation value of Z is 1.
      pauli = np.array([2, 2])
      sub_indices = [0, 1]
      expected_ev = 1.
      actual_ev = estimates_utils.estimate_expval_pauli_from_measurements(
          self.bell_state_ds, pauli, sub_indices, estimator='empirical'
      )
      np.testing.assert_allclose(expected_ev, actual_ev)

    with self.subTest('pauli x'):
      # Expectation value of X is 1.
      pauli = np.array([0, 0])
      sub_indices = [0, 1]
      expected_ev = 1.
      actual_ev = estimates_utils.estimate_expval_pauli_from_measurements(
          self.bell_state_ds, pauli, sub_indices, estimator='empirical'
      )
      np.testing.assert_allclose(expected_ev, actual_ev)


  def test_surface_code_stabilizers(self):
    """Test the stabilizer expectations of surface code."""

    with self.subTest('stabilizer z'):
      z_stabilizers = [[0, 1], [1, 2, 4, 5], [3, 4, 6, 7], [7, 8]]
      expected_ev = np.ones((len(z_stabilizers)))
      actual_z_evs = []
      for stabilizer in z_stabilizers:
        pauli = 2. * np.ones_like(stabilizer)
        ev = estimates_utils.estimate_expval_pauli_from_measurements(
            self.surface_code_ds, pauli, stabilizer, estimator='empirical'
        )
        actual_z_evs.append(ev)
      np.testing.assert_allclose(expected_ev, np.asarray(actual_z_evs))

    with self.subTest('stabilizer x'):
      x_stabilizers = [[2, 5], [0, 1, 3, 4], [4, 5, 7, 8], [3, 6]]
      expected_ev = np.array([0.9959432,  0.95537525, 0.97565923, 0.9959432])
      actual_x_evs = []
      for stabilizer in x_stabilizers:
        pauli = np.zeros_like(stabilizer)
        ev = estimates_utils.estimate_expval_pauli_from_measurements(
            self.surface_code_ds, pauli, stabilizer, estimator='empirical'
        )
        actual_x_evs.append(ev)
      np.testing.assert_allclose(expected_ev, np.asarray(actual_x_evs))


if __name__ == '__main__':
  absltest.main()
