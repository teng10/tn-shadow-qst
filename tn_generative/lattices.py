"""Lattice class and lattice helper functions."""
from __future__ import annotations
import dataclasses
import itertools
import math
from typing import Mapping, Tuple

import numpy as np
import shapely

from tn_generative import func_utils


vectorized_method = func_utils.vectorized_method


@dataclasses.dataclass
class Lattice:
  """Lattice class for storing lattice points and performing operations."""
  points: np.ndarray
  decimal_precision: int = 3
  loc_to_idx: Mapping[Tuple[str, ...], int] = dataclasses.field(
      init=False, repr=False)
  n_sites: int = dataclasses.field(init=False)
  ndim: int = dataclasses.field(init=False)

  def __post_init__(self):
    self.n_sites, self.ndim = self.points.shape
    loc_to_idx = {}
    for idx in range(self.n_sites):
      loc_to_idx[self.point_to_key(self.get_point(idx))] = idx
    self.loc_to_idx = loc_to_idx

  @vectorized_method(signature='(),(x)->()')
  def get_idx(self, point: np.ndarray) -> int:
    """Returns lattice index at the `location`."""
    return self.loc_to_idx[self.point_to_key(point.astype(float))]

  def get_point(self, idx: int) -> np.ndarray:
    """Returns lattice coordinates at the `idx`-th site."""
    return self.points[idx]

  def point_to_key(self, r: np.ndarray):
    """Returns a hashable key for the lattice point `r`."""
    return str(tuple(np.round(r, self.decimal_precision)))

  def merge(self, other: Lattice, raise_on_overlap: bool = False) -> Lattice:
    """Returns a new lattice with points from `self` and `other`."""
    def _unique_ordered(arr):
        """Returns unique vectors in `arr` in order of appearance.
        Note: this is important for DMRG in 2 dimension.
        """
        _, unique_indices = np.unique(arr, axis=0, return_index=True)
        return np.sort(unique_indices)

    common_precision = min(self.decimal_precision, other.decimal_precision)
    combined_points = np.concatenate([self.points, other.points])
    combined_points_rounded = np.round(combined_points, common_precision)
    unique_indices = _unique_ordered(combined_points_rounded)
    unique_combined = combined_points[unique_indices]
    if raise_on_overlap and unique_combined.shape != combined_points.shape:
      raise ValueError('Attempting to merge lattice with overlapp.')
    return Lattice(unique_combined, common_precision)

  def shift(self, vector: np.ndarray) -> Lattice:
    """Returns a new lattice shifted by `vector`."""
    new_points = self.points + vector
    return Lattice(new_points, self.decimal_precision)

  def __add__(self, other):
    return self.merge(other)

  def __radd__(self, other):
    if isinstance(other, int) and other == 0:
      return self
    raise NotImplementedError(f'__radd__ not implemented for {type(other)=}')

  def __eq__(self, other):
    """Returns True if `self` and `other` are equal."""
    return (isinstance(other, Lattice)
            and np.array_equal(self.points, other.points))


class RubyLattice(Lattice):
  def __init__(
      self,
      rho: float = np.sqrt(3.),
      a: float = 1. / 4., 
  ):
    """Ruby lattice of lattice constant `a` and rectangular ratio `rho`.
    
    Args:
      rho: aspect ratio of the ruby lattice.
      a: lattice spacing.
    """
    unit_cell_points = a * np.array(
        [[1., 0.], [1. / 2., np.sqrt(3) / 2.], [3. / 2., np.sqrt(3) / 2.],
        [1. / 2., np.sqrt(3) / 2. + rho], [3. / 2., np.sqrt(3) / 2. + rho], 
        [1., np.sqrt(3) + rho]]
    )
    self.unit_cell = Lattice(unit_cell_points)
    self.a1 = np.array([2 * a * rho * np.sqrt(3) / 2. + a, 0.0])
    self.a2 = np.array([
        a * rho * np.sqrt(3) / 2. + a / 2., 
        a * rho * 1. / 2. + a * np.sqrt(3) / 2 + a * rho
    ])

  def get_expanded_lattice(
      self,
      size_x: int,
      size_y: int,        
  ) -> Lattice:
    """Returns a lattice of size `size_x` x `size_y`."""
    expanded_lattice = sum(
        self.unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(size_x), range(size_y))
    )
    return expanded_lattice  


class KagomeLattice(Lattice):
  def __init__(
      self,
      a: float = 1.0,
  ):
    """Kagome lattice of lattice constant `a`."""

    unit_cell_points = a * np.array([
        [0.25, 0], [-0.25, 0.], [0., np.sqrt(3) / 4.]
        ]
    ) 
    self.unit_cell = Lattice(unit_cell_points)
    self.a1 = a * np.array([1.0, 0.0])
    self.a2 = a * np.array([1. / 2., np.sqrt(3.0) / 2.])

  def get_expanded_lattice(
      self,
      size_x: int,
      size_y: int,        
  ) -> Lattice:
    """Returns a lattice of size `size_x` x `size_y`."""
    expanded_lattice = sum(
        self.unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(size_x), range(size_y))
    )
    return expanded_lattice  
  

def get_restricted(
    lattice: Lattice,
    polygon: shapely.geometry.Polygon,
)-> Lattice:
  """Returns `lattice` with points restricted to be inside of a`polygon`.

  Args:
    lattice: input lattice.
    polygon: a shapely polygon specifying restricted boundaries.

  Returns:
    A part of `lattice` that is contains only points within the `polygon`.
  """
  new_points = []
  for point in lattice.points:
    shapely_point = shapely.geometry.Point(point)
    if polygon.contains(shapely_point):
      new_points.append(point)
  return Lattice(np.stack(new_points), lattice.decimal_precision)


def generate_shapely_hexagon(length:float, x: float, y: float,
) -> shapely.geometry.Polygon:
  """Generates hexagon centered on (x, y) using shapely.

  Args:
    length: length of the hexagon's edge.
    x: x-coordinate of the hexagon's center.
    y: y-coordinate of the hexagon's center.

  Returns:
    The polygon containing the hexagon's coordinates.
  """
  vertices = [
      [x + math.cos(math.radians(angle)) * length,
       y + math.sin(math.radians(angle)) * length]
       for angle in range(0, 360, 60)
  ]
  return shapely.geometry.Polygon(vertices)


def generate_shapely_rectangle(width:float, height:float, x: float, y: float,
) -> shapely.geometry.Polygon:
  """Generates rectangle centered on (x, y) using shapely.

  Args:
    width: width of the rectangle's edge.
    height: height of the rectangle's edge.
    x: x-coordinate of the rectangle's center.
    y: y-coordinate of the rectangle's center.

  Returns:
    The polygon containing the rectangle's coordinates.
  """
  vertices = [
      [x + width / 2., y + height / 2.],
      [x + width / 2., y - height / 2.],
      [x - width / 2., y - height / 2.],
      [x - width / 2., y + height / 2.],
  ]
  return shapely.geometry.Polygon(vertices)
