"""Tests for physical_systems."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from tn_generative import physical_systems


class RubyRydbergTests(parameterized.TestCase):
  """Tests for using physical_systems to define ruby rydberg hamiltonian."""

  def setUp(self):
    """Set up using ruby rydberg vanderwaals hamiltonian as an example."""
    self.ruby_system = physical_systems.RubyRydbergPXP(2, 2)

  @parameterized.parameters(
    {'Lx': 2, 'Ly': 2},
    {'Lx': 3, 'Ly': 3},
    {'Lx': 4, 'Ly': 3},
    {'Lx': 5, 'Ly': 3},
  )
  def test_lattice(self):
    """Test that the lattice can be generated."""
    self.assertTrue(hasattr(self.ruby_system , '_lattice'))

  def test_bonds(self):
    """Test that nearest neighbour bonds are generated correctly."""
    bonds = self.ruby_system._get_nearest_neighbour_bonds()
    nn_expected_nodes = np.array([
       [ 0,  1], [ 0,  2], [ 1,  2], [ 3,  4], [ 3,  5], [ 4,  5], [ 6,  7],
       [ 6,  8], [ 7,  8], [ 9, 10], [ 9, 11], [10, 11], [12, 13], [12, 14],
       [13, 14], [15, 16], [15, 17], [16, 17], [18, 19], [18, 20], [19, 20],
       [21, 22], [21, 23], [22, 23]
    ])
    np.testing.assert_allclose(bonds[0].nodes, nn_expected_nodes)

  @parameterized.parameters(
    {'Lx': 2, 'Ly': 2},
    {'Lx': 3, 'Ly': 2},
    {'Lx': 4, 'Ly': 2},
    {'Lx': 5, 'Ly': 2},
  )
  def test_boundary_field(self, Lx, Ly):
    """."""
    total_unit_cells = Lx * Ly
    sites_unit_cell = 6
    ruby_system = physical_systems.RubyRydbergPXP(Lx, Ly, boundary='periodic')
    expected_boundary_sites = np.array([
        3, 5, 9, 11,
        (total_unit_cells - 1) * sites_unit_cell,
        (total_unit_cells - 1) * sites_unit_cell + 2,
        (total_unit_cells - 2) * sites_unit_cell,
        (total_unit_cells - 2) * sites_unit_cell + 2,
    ])
    actual_boundary_sites = ruby_system.boundary_sites
    np.testing.assert_allclose(actual_boundary_sites, expected_boundary_sites)

if __name__ == '__main__':
  absltest.main()
