"""Tests for estimates_utils.py."""
import json
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
from tn_generative import mps_utils
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
        sub_rdm = (mps.partial_trace(sub_indices, rescale_sites=False)).to_dense()
        actual_ev = np.trace(sub_rdm @ subsystem_mpo.to_dense())
        # actual_ev = (sub_rdm.apply(subsystem_mpo)).trace()
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


class ConstructReducedDesnityMat(parameterized.TestCase):
  """Tests for constructions of reduced density matrices."""

  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/example_dataset.nc')
    test_ds = xr.load_dataset(ds_path)
    full_test_ds = data_utils.combine_complex_ds(test_ds)
    self.test_ds = full_test_ds.sel(sample=slice(0, 499))
    self.mps = mps_utils.xarray_to_mps(self.test_ds)
  
  def test_reduced_density_matrices_surface_code(self, ):
    """Tests that reduced density matrix constructed from dataset."""
    subsystems = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]]
    for subsystem in subsystems:
      with self.subTest(f'{subsystem=}'):
        rdm_mpo = estimates_utils._construct_reduced_density_matrix(
            self.test_ds, subsystem
        )
        expected_rdm = (self.mps.partial_trace(
            subsystem, rescale_sites=False
        )).to_dense()
        np.testing.assert_allclose(expected_rdm, rdm_mpo)

  def test_reduced_density_matrices_ghz(self):
    """Tests reduced density matrix for ghz states."""
    bitstring = np.array([0, 1, 0, 0, 0])
    mps_prod = qtn.MPS_computational_state(bitstring)
    subsystem = [1, 2]
    expected_dm = mps_prod.partial_trace(subsystem).to_dense()
    bitstring_subsys = bitstring[np.array(subsystem)]
    mps_subsystem = qtn.MPS_computational_state(bitstring_subsys)
    bitstring_dense = mps_subsystem.to_dense()
    bitstring_dm = np.outer(bitstring_dense, np.conj(bitstring_dense))
    np.testing.assert_allclose(expected_dm, bitstring_dm)

if __name__ == '__main__':
  absltest.main()
