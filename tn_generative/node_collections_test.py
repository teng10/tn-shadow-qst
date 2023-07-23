"""Tests for node_collections."""
import itertools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import matplotlib.pyplot as plt

from tn_generative import node_collections
from tn_generative import lattices
from tn_generative import plotting_utils


class NodesTests(parameterized.TestCase):
  """Tests for using node_collections."""

  def setUp(self):
    self.a1 = np.array([2.0, 0.0])
    self.a2 = np.array([1.0, np.sqrt(3.0)])
    self.unit_cell_points = np.stack([
      np.array([0.5, 0]), np.array([-0.5, 0.]), np.array([0., np.sqrt(3) / 2.])
      ]
    )
    self.nx, self.ny = 5, 4
    unit_cell = lattices.Lattice(self.unit_cell_points)
    expanded_lattice = sum(
        unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(self.nx), range(self.ny))
    )
    hexagon = lattices.generate_shapely_hexagon(3.1, 6, 3 * np.sqrt(3)/2, )
    self.expanded_lattice = lattices.get_restricted(expanded_lattice, hexagon)

  def test_nearest_neighbors(self):
    """Tests that the nearest neighbors are correct."""
    bonds = node_collections.get_nearest_neighbors(self.expanded_lattice, 1.1)
    print(bonds.lattice.points.shape)
    expected_nodes = np.array([[ 0, 1], [ 0, 2], [ 1, 3], [ 1, 6], [ 2, 4],
        [2, 7], [3, 5], [3,  6], [4, 7], [4, 8], [5, 10], [6, 9], [6, 11],
        [7, 9], [7, 12], [8, 13], [9, 11], [9, 12], [10, 14], [10, 16],
        [11, 14], [11, 17], [12, 15], [12, 18], [13, 15], [13, 19], [14, 16],
        [14, 17], [15, 18], [15, 19], [16, 21], [17, 20], [17, 22], [18, 20],
        [18, 23], [19, 24], [20, 22], [20, 23], [21, 25], [22, 25], [22, 27],
        [23, 26], [23, 28], [24, 26], [25, 27], [26, 28], [27, 29], [28, 29]]
    )
    np.testing.assert_allclose(expected_nodes, bonds.nodes)

  def test_coords_idx_conversion(self):
    """Tests that the idx_to_point and point_to_idx functions are inverses."""
    bonds = node_collections.get_nearest_neighbors(self.expanded_lattice, 1.1)
    np.testing.assert_allclose(
        np.array([5, 4]),
        bonds.idx_from_coords(bonds.coords_from_idx(np.array([5, 4])))
    )
    fig, ax = plt.subplots(1, 1)
    plotting_utils.plot_lattice(self.expanded_lattice, ax, True)
    plotting_utils.plot_bonds(bonds, ax, plot_with_arrows=True)
    ax.set_aspect("equal")

  def test_loops(self):
    """Tests by running examples of loop functionalities."""
    loop_nodes = np.array(
        [[6, 11, 11, 17, 17, 14, 14, 11, 11, 9, 9, 6]], dtype=np.int64
    )
    base_loop = node_collections.NodesCollection(loop_nodes,
        self.expanded_lattice
    )

    with self.subTest("Base loop"):
      np.testing.assert_allclose(loop_nodes, base_loop.nodes)
      fig, ax = plt.subplots(1, 1)
      plotting_utils.plot_lattice(self.expanded_lattice, ax, True)
      plotting_utils.plot_loops(base_loop, ax, plot_with_arrows=True)
      ax.set_aspect("equal")

    with self.subTest("Shifted loop"):
      fig, ax = plt.subplots(1, 1)
      plotting_utils.plot_lattice(self.expanded_lattice, ax, True)
      plotting_utils.plot_loops(
          base_loop + base_loop.shift(np.array([1., 1.732])), ax
      )  # example of combinging
      ax.set_aspect("equal")

    with self.subTest("Tile on lattice"):
      fig, ax = plt.subplots(1, 1)
      a1, a2 = np.array([-2., 0.0]),  np.array([1., 1.732])
      plotting_utils.plot_lattice(self.expanded_lattice, ax)
      plotting_utils.plot_loops(
          node_collections.tile_on_lattice(base_loop, a1, a2, 8),
          ax, plot_with_arrows=True
      )
      ax.set_aspect("equal")

if __name__ == "__main__":
  absltest.main()
