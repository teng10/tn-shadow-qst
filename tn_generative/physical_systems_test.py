"""Tests for physical_systems."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from tn_generative import physical_systems


class RubyRydbergTests(parameterized.TestCase):
  """Tests for using physical_systems to define ruby rydberg hamiltonian."""

  def setUp(self):
    """Set up using ruby rydberg vanderwaals hamiltonian as an example."""
    self.ruby_system = physical_systems.RubyRydbergVanderwaals(2, 2)

  @parameterized.parameters(
    dict(Lx=2, Ly=2),
    dict(Lx=3, Ly=3),
    dict(Lx=3, Ly=4),
    dict(Lx=3, Ly=5)
  )
  def test_lattice(self, Lx, Ly):
    """Test that the lattice can be generated."""
    ruby_general_system = physical_systems.RubyRydbergVanderwaals(Lx, Ly)
    ruby_general_system._lattice

  def test_bonds(self):
    """Test that nearest neighbour bonds are generated correctly."""
    bonds = self.ruby_system._get_nearest_neighbour_bonds()
    nn_expected_nodes = np.array([
      [ 0,  2], [ 0,  4], [ 1,  3], [ 1,  5], [ 2,  4], [ 3,  5], [ 6,  8],
      [ 6, 10], [ 7,  9], [ 7, 11], [ 8, 10], [ 9, 11], [12, 14], [12, 16], 
      [13, 15], [13, 17], [14, 16], [15, 17], [18, 20], [18, 22], [19, 21],
      [19, 23], [20, 22], [21, 23]
    ])
    np.testing.assert_allclose(bonds[0].nodes, nn_expected_nodes)


if __name__ == "__main__":
  absltest.main()
