"""Tests for shadow_utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import os

import numpy as np
import xarray as xr
from pennylane import ClassicalShadow

from tn_generative import data_utils
from tn_generative import shadow_utils


class ConstructShadows(parameterized.TestCase):
  """Tests for constructions of shadow states."""

  def setUp(self):
    script_path = os.path.abspath(__file__)
    # Extract the directory containing the script
    current_file_dir = os.path.dirname(script_path)
    ds_path = os.path.join(current_file_dir, 'test_data/bell_state_ds.nc')
    bell_state_ds = xr.load_dataset(ds_path)
    self.bell_state_ds = data_utils.combine_complex_ds(bell_state_ds)
    single_shot_fn = shadow_utils._get_shadow_single_shot_fn(self.bell_state_ds)
    self.shadow_state_custom = shadow_utils.construct_subsystem_shadows(
      bell_state_ds, [0, 1], single_shot_fn)

  #TODO(YT): add meaningful tests for surface code.
  # def test_subsystem_shadows_surface_code(self, ):
  #   """Test subsystem shadows constructed from dataset."""
  #   subsystems = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]]
  #   for subsystem in subsystems:
  #     with self.subTest(f'{subsystem=}'):
  #       single_shot_fn = shadow_utils._get_shadow_single_shot_fn(self.test_ds)
  #       rdm_mpo = shadow_utils._construct_subsystem_shadows(
  #           self.test_ds, subsystem, single_shot_fn
  #       )
  #       expected_rdm = (self.mps.partial_trace(
  #           subsystem, rescale_sites=False
  #       )).to_dense()
  #       np.testing.assert_allclose(expected_rdm, rdm_mpo)

  def test_bell_state_reconstruction(self):
    """Test reconstruction of bell state."""
    bell_state = np.array(
        [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
    )
    np.allclose(bell_state, self.shadow_state_custom, atol=1e-1)

  def test_bell_state_pennylane(self):
    """Test reconstruction of bell state."""
    shadow = ClassicalShadow(
        self.bell_state_ds['measurement'].values,
        self.bell_state_ds['basis'].values
    )
    shadow_state = np.mean(shadow.global_snapshots(), axis=0)
    np.allclose(shadow_state, self.shadow_state_custom, atol=1e-1)

  # TODO(YT): finish ghz test.
  # def test_reduced_density_matrices_ghz(self):
  #   """Tests reduced density matrix for bell states."""
  #   bitstring = np.array([0, 1, 0, 0, 0])
  #   mps_prod = qtn.MPS_computational_state(bitstring)
  #   subsystem = [1, 2]
  #   expected_dm = mps_prod.partial_trace(subsystem).to_dense()
  #   bitstring_subsys = bitstring[np.array(subsystem)]
  #   mps_subsystem = qtn.MPS_computational_state(bitstring_subsys)
  #   bitstring_dense = mps_subsystem.to_dense()
  #   bitstring_dm = np.outer(bitstring_dense, np.conj(bitstring_dense))
  #   np.testing.assert_allclose(expected_dm, bitstring_dm)


if __name__ == '__main__':
  absltest.main()
