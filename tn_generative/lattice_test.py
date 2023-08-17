"""Tests for lattices.py."""
import itertools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import matplotlib.pyplot as plt

from tn_generative import lattices
from tn_generative import plotting_utils


class KagomeLatticeTests(parameterized.TestCase):
  """Tests for using lattices creating kagome lattice."""

  def setUp(self):
    """Set up using kagome lattice as an example."""
    self.a1 = np.array([2.0, 0.0])
    self.a2 = np.array([1.0, np.sqrt(3.0)])
    self.unit_cell_points = np.stack([
      np.array([0.5, 0]), np.array([-0.5, 0.]), np.array([0., np.sqrt(3) / 2.])
      ]
    )
    self.nx, self.ny = 8, 8
    self.unit_cell = lattices.Lattice(self.unit_cell_points)

  def test_kagome_lattice(self):  #TODO(YT): move all plotting to colab.
    """Test plotting the kagome lattice."""
    with self.subTest("Unit cell"):
      unit_cell = lattices.Lattice(self.unit_cell_points)
      expected_cell = lattices.Lattice(
          points=np.array([
          [ 0.5  ,  0.   ],
          [-0.5  ,  0.   ],
          [ 0.   ,  np.sqrt(3) / 2.]]),
      )
      self.assertTrue(unit_cell == expected_cell)
      fig, ax = plt.subplots(1, 1)
      plotting_utils.plot_lattice(unit_cell, ax, annotate=True)
      ax.set_aspect("equal")

    with self.subTest("Expanded lattice"):
      expanded_lattice = sum(
          unit_cell.shift(self.a1 * i + self.a2 * j)
          for i, j in itertools.product(range(self.nx), range(self.ny))
      )     
      fig, ax = plt.subplots(1, 1)
      plotting_utils.plot_lattice(expanded_lattice, ax, annotate=True)
      ax.set_aspect("equal")

    with self.subTest("Restricted lattice"):
      hexagon = lattices.generate_shapely_hexagon(6.1, 10.0, 7 * np.sqrt(3)/2)
      la = lattices.get_restricted(expanded_lattice, hexagon)
      fig, ax = plt.subplots(1, 1)
      plotting_utils.plot_lattice(la, ax, annotate=True)
      ax.set_aspect("equal")

  def test_idx_point_conversion(self):
    """Tests that the idx_to_point and point_to_idx functions are inverses."""
    unit_cell = self.unit_cell
    indices = unit_cell.get_idx(
        unit_cell.get_point(np.arange(unit_cell.n_sites))
    )
    np.testing.assert_allclose(
        np.array([1]),
        np.squeeze(unit_cell.get_idx([unit_cell.get_point([1])]))
    )

  @parameterized.parameters(2, 3, 4, 5, 6, 7)
  def test_shifted_coordinate(self, shift):
    """Tests that the shifted point is numerically precise."""
    test_point  = self.unit_cell.points[2]
    test_point_shifted = test_point + self.a2 * shift
    unit_cell_shifted = self.unit_cell.shift(self.a2 * shift)
    np.testing.assert_allclose(
        test_point_shifted, unit_cell_shifted.points[2]
    )    

if __name__ == "__main__":
  absltest.main()
