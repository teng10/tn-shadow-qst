"""Nodes collection and loops"""
from __future__ import annotations
import dataclasses
from typing import Mapping, Tuple

import numpy as np
import einops
import scipy.spatial as sp_spatial

# from tn_generative import typing
from tn_generative import lattice
from tn_generative import func_utils

# Array = typing.Array  #TODO(YT): probably remove this because only need np.ndarray
Lattice = lattice.Lattice
vectorized_method = func_utils.vectorized_method


@dataclasses.dataclass
class NodesCollection:
  nodes: np.ndarray
  lattice: Lattice
  tuple_to_idx: Mapping[Tuple[int, ...], int] = dataclasses.field(
      init=False, repr=False)
  count: int = dataclasses.field(init=False)
  length: int = dataclasses.field(init=False)

  def __post_init__(self):
    tuple_to_idx = {}
    for idx, path in enumerate(self.nodes):
      tuple_to_idx[tuple(path)] = idx
    self.tuple_to_idx = tuple_to_idx
    self.count, self.length = self.nodes.shape

  @vectorized_method(signature='(),(x)->()')
  def idx_from_sites(self, sites):
    return self.tuple_to_idx[tuple(sites)]

  @vectorized_method(signature='(),(x, y)->()')
  def idx_from_coords(self, coords):
    return self.idx_from_sites(self.lattice.get_idx(coords))

  def sites_from_idx(self, idx):
    return self.nodes[idx]

  def coords_from_idx(self, idx):
    return self.lattice.points[self.nodes[idx]]

  def shift(self, vector: np.ndarray) -> NodesCollection:
    nodes_coords = self.coords_from_idx(np.arange(self.count))
    nodes_coords += vector  # shift coordinates of all path vertices.
    return self.from_coords(nodes_coords, self.lattice)

  def merge(
      self,
      other: NodesCollection,
      raise_on_duplicates: bool = False,
  ) -> NodesCollection:
    if self.lattice != other.lattice:
      raise ValueError('Cannot merge nodes from different lattices.')
    combined_nodes = np.concatenate([self.nodes, other.nodes])
    unique_nodes = np.unique(combined_nodes, axis=0)
    if raise_on_duplicates and unique_nodes.shape != combined_nodes.shape:
      raise ValueError(f'Attempting to merge overlapping nodes {self=}{other=}')
    return self.__class__(unique_nodes, self.lattice)

  def __add__(self, other):
    return self.merge(other)

  def __radd__(self, other):
    if isinstance(other, int) and other == 0:
      return self
    raise NotImplementedError(f'__radd__ not implemented for {type(other)=}')

  @classmethod  # Question: why is this a class method, different from others?
  def from_coords(
      cls: NodesCollection,
      path_coords: np.ndarray,
      lattice: Lattice,
  ) -> NodesCollection:  # TODO(YT): this is correct???
    paths = lattice.get_idx(path_coords)
    return cls(paths, lattice)


def extract_edge_sequence(nodes_collection):
  """Extracts sequence of edges from `nodes_collection`."""
  if (nodes_collection.length % 2) != 0:
    raise ValueError(f'Cannot interpret {nodes_collection.length=} as edges.')
  return einops.rearrange(nodes_collection.nodes, '... (n b) -> ... n b', b=2)


def extract_edge_coords(nodes_collection):
  """Extracts coords of sequence of edges from `nodes_collection`."""
  if (nodes_collection.length % 2) != 0:
    raise ValueError(f'Cannot interpret {nodes_collection.length=} as edges.')
  all_indices = np.arange(nodes_collection.count)
  nodes_coords = nodes_collection.coords_from_idx(all_indices)
  return einops.rearrange(nodes_coords, 'a (n b) c -> (a n) b c', b=2, c=2)


def tile_on_lattice(base_loop, a1, a2, n_steps):
  """Tiles `base_loop` on the lattice with `a1`, `a2` shifts."""
  all_loops = []
  for i in range(-n_steps, n_steps):
    for j in range(-n_steps, n_steps):
      try:
        new_loop = base_loop.shift(a1 * i + a2 * j)
        all_loops.append(new_loop)
      except KeyError:
        ...
  return sum(all_loops)


def get_nearest_neighbors(
    lattice: Lattice, 
    nb_radius: float,
    ) -> NodesCollection:
  """Returns neighbors of `lattice` nodes within distance `nb_radius`."""
  # TODO(YT): define `close_pairs_pdist` in `lattice.py`. 
  def condensed_to_pair_indices(n,k):
    x = n-(4.*n**2-4*n-8*k+1)**.5/2-.5
    i = x.astype(int)
    j = k+i*(i+3-2*n)/2+1
    return i.astype(int),j.astype(int)

  def close_pairs_pdist(X, max_d):
    d = sp_spatial.distance.pdist(X)
    k = (d<max_d).nonzero()[0]
    return condensed_to_pair_indices(X.shape[0],k)   
  
  pairwise_distance = close_pairs_pdist(lattice.points, nb_radius)
  nodes_coords = lattice.points[np.stack(pairwise_distance).T]
  return NodesCollection.from_coords(nodes_coords, lattice)


def build_path_from_sequence(
    indices: np.ndarray,
    lattice: Lattice,
) -> NodesCollection:
  """Builds path sequence from ordered `indices`."""
  if indices.ndim != 1:
    raise ValueError(f'Expected 1d array of indices, got {indices.ndim=}')
  path = np.empty((indices.size * 2,), dtype=indices.dtype)
  path[0::2] = indices
  path[1::2] = np.roll(indices, shift=-1)
  return NodesCollection(path[np.newaxis, :], lattice)
