"""Tests for estimates_utils.py."""
import json
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen

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
      sub_rdm = mps.partial_trace(sub_indices)
      actual_ev = (sub_rdm.apply(subsystem_mpo)).trace()
      np.testing.assert_allclose(expected_ev, actual_ev)


if __name__ == '__main__':
  absltest.main()
