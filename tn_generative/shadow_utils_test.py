"""Tests for shadow_utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import os

import numpy as np
import quimb.tensor as qtn
import quimb.gen as qugen
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

  # #TODO(YT): add meaningful tests for surface code.
  # Would have to use pauli shadow, need `surface_code_x_y_z_data.nc` file.
  # # Satbilizers are alredy checked in estimate_utils_test.py
  # def test_stabilizers_shadows_surface_code(self, ):
  #   """Test subsystem shadows constructed from dataset."""
  #   ds_path = os.path.join(
  #       current_file_dir, 'test_data/surface_code_x_y_z_data.nc'
  #   )
  #   surface_code_ds = xr.load_dataset(ds_path)
  #   self.ds_sc = data_utils.combine_complex_ds(surface_code_ds)
  #   mps = mps_utils.xarray_to_mps(self.ds_sc)
  #   subsystems = [[0, 1, 3, 4], [4, 5, 7, 8]]
  #   for subsystem in subsystems:
  #     with self.subTest(f'{subsystem=}'):
  #       rdm_mpo = shadow_utils._construct_subsystem_shadows(
  #           self.ds_sc, subsystem, single_shot_fn
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

  # # TODO(YT): finish random state test with pauli shadow
  # def test_pauli_shadows(self):
  #   """Tests reduced density matrix for random states."""
  #   qugen.rand.seed_rand(self.seed)
  #   mps = qtn.MPS_rand_state(6, 2)
  #   # TODO (YT): generate dataset from this random state
  #   ds = run_data_generation.generate_data_from_mps(
  #       mps, ds_size, measurement_scheme
  #   )
  #   subsystem_size = 3
  #   for subsystem in itertools.combinations(range(6), subsystem_size):
  #     with self.subTest(f'{subsystem=}'):
  #       single_shot_fn = shadow_utils.shadow_pauli_single_shot_vectorized
  #       bitstring_dm = shadow_utils.construct_subsystem_shadows(
  #           ds, subsystem, single_shot_fn
  #       )
  #       expected_dm = mps.partial_trace(subsystem).to_dense()
  #       np.testing.assert_allclose(expected_dm, bitstring_dm)


if __name__ == '__main__':
  absltest.main()
