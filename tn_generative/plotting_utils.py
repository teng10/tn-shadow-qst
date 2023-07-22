#@title ||  Helper function for plotting. 

from tn_generative import lattice
from tn_generative import node_collections


def plot_lattice(la: lattice.Lattice, ax, annotate: bool = False):
  ax.scatter(*la.points.T)
  if annotate:
    for i in range(la.n_sites):
      ax.annotate(str(i), la.points[i])


def plot_bonds(nodes_collection, ax, c='r', lw=3, plot_with_arrows: bool=False):
  if nodes_collection.length != 2:
    raise ValueError(f'Expected {nodes_collection.length=} to be equal to 2.')
  for i in range(nodes_collection.count):
    bond_vector = nodes_collection.coords_from_idx(i)
    if plot_with_arrows:
      dx = bond_vector[1, 0] - bond_vector[0, 0]
      dy = bond_vector[1, 1] - bond_vector[0, 1]
      ax.arrow(bond_vector[0, 0], bond_vector[0, 1],
               dx=dx, dy=dy, width=0.07, head_width=0.5,
               length_includes_head=True)
    else:
      ax.plot(bond_vector[:, 0], bond_vector[:, 1], c=c, lw=lw)


def plot_loops(nodes_collection, ax, c='r', lw=2, plot_with_arrows: bool=False):
  edge_coords = node_collections.extract_edge_coords(nodes_collection)
  for i, edge in enumerate(edge_coords):
    if plot_with_arrows:
      dx = edge[1, 0] - edge[0, 0]
      dy = edge[1, 1] - edge[0, 1]
      ax.arrow(edge[0, 0], edge[0, 1],
               dx=dx, dy=dy, width=0.07, head_width=0.5,
               length_includes_head=True)
    else:
      ax.plot(edge[:, 0], edge[:, 1], c=c, lw=lw)
