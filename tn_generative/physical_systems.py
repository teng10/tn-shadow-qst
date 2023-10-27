"""Physical systems for generating dataset."""
import abc
from abc import abstractmethod
import itertools
import functools
from typing import Callable

import numpy as np
import einops
import quimb.tensor as qtn
import quimb.experimental.operatorbuilder as quimb_exp_op

from tn_generative import node_collections
from tn_generative import lattices
from tn_generative import types


def _distance_fn_mod(
    x: np.ndarray , y: np.ndarray, mod_vector: np.ndarray
) -> float:
  """Distance function for periodic boundary conditions."""
  shifted_distances = [
      np.sqrt(((x - y + i * mod_vector)**2).sum()) for i in range(-1, 2)
  ]
  return min(shifted_distances)


class PhysicalSystem(abc.ABC):
  """Abstract class for defining physical systems."""

  @property
  def hilbert_space(self) -> types.HilbertSpace | None:
    """Returns hilbert space of the physical system."""
    return None

  @abstractmethod
  def get_terms(self) -> types.TermsTuple | None:
    """Returns list of terms in the hamiltonian."""
    return None

  @abstractmethod
  def get_ham(self) -> qtn.MatrixProductOperator:
    """Returns a hamiltonian MPO."""

  def get_sparse_operator(
      self, 
      terms: types.TermsTuple,
  ) -> quimb_exp_op.SparseOperatorBuilder:
    """Generates operator including `terms` using sparse operator builder."""
    if self.hilbert_space is None:
      raise ValueError(
          f'subclass {self.__name__} did not implement custom `hilbert_space`.'
      )    
    sparse_operator = quimb_exp_op.SparseOperatorBuilder(
        hilbert_space=self.hilbert_space
    )    
    for term in terms:  # add all terms to the operator.
      sparse_operator += term
    return sparse_operator
  
  def get_ham_mpos(self) -> list[qtn.MatrixProductOperator]:
    """Returns MPOs for list of terms in the hamiltonian.

    Note: this method returns terms in `get_terms` method with +1 coupling.

    Returns:
      List of MPOs in the hamiltonian.
    """
    if self.get_terms() is None:
      raise ValueError(
          f'subclass {self.__name__} did not implement custom `get_terms`.'
          f'subclass {self.__name__} should either implement custom' 
          '`get_ham_mpos` or provide `hilbert_space` and implement `get_terms`.'          
      )
    mpos = []
    terms = [(1., *term[1:]) for term in self.get_terms()]
    for term in terms:
      mpos.append(
          self.get_sparse_operator([term]).build_mpo()
      )
    return mpos

  def get_obs_mpos(
      self,
      terms: types.TermsTuple,
  ) -> list[qtn.MatrixProductOperator]:
    """Get observables `terms` as MPOs.

    Args:
      terms: list of terms to include in the MPOs.

    Returns:
      List of MPOs.
    """
    mpos = []
    for term in terms:
      mpos.append(
          self.get_sparse_operator([term]).build_mpo()
      )
    return mpos


class SurfaceCode(PhysicalSystem):
  """Implementation for surface code.
    Note: this constructor assumes default ferromagnetic coupling. i.e.
    `coupling_value == 1` corresponds to `H = -1 * (Σ A) ...`.
    Note: `get_terms` method in this constructor assumes +1 coupling.
  """

  def __init__(
      self,
      Lx: int,
      Ly: int,
      coupling_value: float = 1.0,
      onsite_z_field: float = 0.,
  ):
    self.n_sites = int(Lx * Ly)
    self.Lx = Lx
    self.Ly = Ly
    self.coupling_value = coupling_value
    self.onsite_z_field = onsite_z_field

  @property
  def hilbert_space(self) -> types.HilbertSpace:
    return quimb_exp_op.HilbertSpace(self.n_sites)

  def _get_surface_code_collections(
      self,
  ) -> tuple[node_collections.NodesCollection, ...]:
    """Constructs `NodeCollection`s for all terms in surface code Hamiltonian.
    """
    if self.Lx % 2 != 1 or self.Ly % 2 !=1:
      raise NotImplementedError('Only odd system sizes are implemented.')
    a1 = np.array([1.0, 0.0])  # unit vectors for square lattice.
    a2 = np.array([0.0, 1.0])
    unit_cell_points = np.stack([np.array([0, 0])])
    unit_cell = lattices.Lattice(unit_cell_points)

    lattice = sum(
        unit_cell.shift(a1 * i + a2 * j)
        for i, j in itertools.product(range(self.Lx), range(self.Ly))
    )
    # generating terms with Z plaquetts at the top left corner.
    z_top_loc = np.array((0, self.Ly - 1))
    z_base_plaquette = node_collections.build_path_from_sequence(
        lattice.get_idx(
            np.stack([  # specify plaquet going right, down, right and back up.
                z_top_loc, z_top_loc + a1, z_top_loc + a1 - a2, z_top_loc - a2
                ])
        ),
        lattice
    )
    # fixing boundary terms with x-x interactions
    # for Z plaquetts on the boundary.
    x_top_boundary = node_collections.build_path_from_sequence(
        lattice.get_idx(np.stack([z_top_loc, z_top_loc + a1])), lattice
    )
    x_bot_boundary = x_top_boundary.shift(-a2 * (self.Ly - 1) + a1)

    # generating terms with X plaquetts at the second square
    # from top left corner.
    x_top_loc = z_top_loc + a1
    x_base_plaquette = node_collections.build_path_from_sequence(
        lattice.get_idx(
            np.stack([
                x_top_loc, x_top_loc + a1, x_top_loc + a1 - a2, x_top_loc - a2
            ])
        ),
        lattice
    )
    # fixing boundary terms with zz interactions on the boundary.
    z_left_boundary = node_collections.build_path_from_sequence(
        lattice.get_idx(np.stack([z_top_loc - a2, z_top_loc - 2 * a2])), lattice
    )
    z_right_boundary = z_left_boundary.shift(a1 * (self.Lx - 1) + a2)

    n_shifts = max(self.Lx, self.Ly)
    z_plaquettes = node_collections.tile_on_lattice(
        z_base_plaquette, 2 * a1, a1 + a2, n_shifts
    )
    x_plaquettes = node_collections.tile_on_lattice(
        x_base_plaquette, 2 * a1, a1 + a2, n_shifts
    )
    x_boundaries = (
        node_collections.tile_on_lattice(
            x_top_boundary, 2 * a1, 0 * a2, n_shifts) +
        node_collections.tile_on_lattice(
            x_bot_boundary, 2 * a1, 0 * a2, n_shifts)
    )
    z_boundaries = (
        node_collections.tile_on_lattice(
            z_left_boundary, 0 * a1, 2 * a2, n_shifts) +
        node_collections.tile_on_lattice(
            z_right_boundary, 0 * a1, 2 * a2, n_shifts)
    )
    return z_plaquettes, x_plaquettes, z_boundaries, x_boundaries

  def get_terms(self) -> types.TermsTuple:
    """Generates all stabilizer terms in the surface code
    from 4-plaquettes and 2-site boundaries.

    z_plaquettes: collection of 4-sites forming σz plaquettes.
    x_plaquettes: collection of 4-sites forming σx plaquettes.
    x_boundaries: collection of 2 boundary sites on Z plaquettes with x-x term.
    z_boundaries: collection of 2 boundary sites on X vertices with z-z term.

    Args:
      Lx: number of sites in x direction.
      Ly: number of sites in y direction.
      coupling_value: coupling coefficient.

    Returns:
      List of terms `z_plaquettes, x_plaquettes, z_boundaries, x_boundaries`
      representing the surface code hamiltonian.
    """
    z_plaquettes, x_plaquettes, z_boundaries, x_boundaries = (
        self._get_surface_code_collections()
    )
    # extract edges specified by lattice sites for each interaction term.
    z_plaquett_edges = node_collections.extract_edge_sequence(z_plaquettes)
    x_plaquett_edges = node_collections.extract_edge_sequence(x_plaquettes)
    xx_boundary_edges = node_collections.extract_edge_sequence(x_boundaries)
    zz_boundary_edges = node_collections.extract_edge_sequence(z_boundaries)
    # get sites for each interaction term by selecting sites.
    z_plaquett_sites = np.unique(
        einops.rearrange(z_plaquett_edges, 'b i s -> b (i s)'), axis=-1)
    x_plaquett_sites = np.unique(
        einops.rearrange(x_plaquett_edges, 'b i s -> b (i s)'), axis=-1)
    z_boundary_sites = np.unique(
        einops.rearrange(zz_boundary_edges, 'b i s -> b (i s)'), axis=-1)
    x_boundary_sites = np.unique(
        einops.rearrange(xx_boundary_edges, 'b i s -> b (i s)'), axis=-1)
    all_terms = []
    for i, j, k, l in z_plaquett_sites:
      terms = -self.coupling_value, ('z', i), ('z', j), ('z', k), ('z', l)
      all_terms.append(terms)
    for i, j, k, l in x_plaquett_sites:
      terms = -self.coupling_value, ('x', i), ('x', j), ('x', k), ('x', l)
      all_terms.append(terms)
    for i, j in z_boundary_sites:
      terms = -self.coupling_value, ('z', i), ('z', j)
      all_terms.append(terms)
    for i, j in x_boundary_sites:
      terms = -self.coupling_value, ('x', i), ('x', j)
      all_terms.append(terms)
    if self.onsite_z_field != 0.:  # add onsite fields if specified.
      for site in range(self.Lx * self.Ly):
        all_terms.append((-self.onsite_z_field, ('z', site)))
    return all_terms

  def get_ham(self) -> qtn.MatrixProductOperator:
    """Get surface code hamiltonian as MPO."""
    surface_code_ham = self.get_sparse_operator(self.get_terms())
    return surface_code_ham.build_mpo()


class RubyRydberg(PhysicalSystem):  #TODO(YT): add tests.
  """Implementation for ruby Rydberg hamiltonian.

    Note: this constructor assumes Van der Waals interactions among neibours. 
    The range of neibours are determined by Callable `nb_ratio_fn`, depending on 
    ruby lattice aspect ratio, `rho`specified in ascending order.

    Args:
      Lx: number of unit cells in x direction.
      Ly: number of unit cells in y direction.
      delta: detuning of the laser from the atomic transition, `z` field.
      rho: aspect ratio of the ruby lattice.
      rb: Rydberg blockade radius, in units of lattice spacing.
      omega: laser Rabi frequency, `x` field.
      nb_ratio_fn: Callable that returns a tuple of ascending neibour radii.  

    Returns:
      Ruby Rydberg hamiltonian Physical system.
  """

  def __init__(
      self,
      Lx: int,
      Ly: int,
      delta: float = 5.0,
      boundary_z_field: float = 0.,
      rho: float = np.sqrt(3.),  
      rb: float = 3.8,  
      omega: float = 1.,
      nb_ratio_fn: Callable[[float], tuple[float, ...]] = lambda rho: (
          1., rho, np.sqrt(1. + rho**2)
      ),
      boundary: str = 'open',
  ):
    self.n_sites = int(Lx * Ly * 6)
    self.Lx = Lx
    self.Ly = Ly
    self.delta = delta
    self.boundary_z_field = boundary_z_field
    self.a = 1. / 4.  # lattice spacing.
    self.omega = omega
    self.rho = rho  
    self.epsilon = 1e-3
    self.nb_radii = tuple(r * self.a + self.epsilon for r in nb_ratio_fn(self.rho))
    self.ruby_lattice = lattices.RubyLattice(rho=self.rho, a=self.a)
    self._lattice = self.ruby_lattice.get_expanded_lattice(self.Lx, self.Ly)
    self.boundary = boundary
    self.boundary_sites = []  # sites on the boundary.
    if self.boundary == 'periodic':
      total_unit_cells = self.Lx * self.Ly
      sites_unit_cell = 6
      self.boundary_sites = [
          3, 5, 9, 11,
          (total_unit_cells - 2) * sites_unit_cell,
          (total_unit_cells - 2) * sites_unit_cell + 2,
          (total_unit_cells - 1) * sites_unit_cell,
          (total_unit_cells - 1) * sites_unit_cell + 2,
      ]
  
  @property
  def vs(self) -> np.ndarray:
    return self._vs
  
  @property
  def hilbert_space(self) -> types.HilbertSpace:
    return quimb_exp_op.HilbertSpace(self.n_sites)
  

  def _get_annulus_bonds(
      self,
      nb_outer: float,
      nb_inner: float = 0.,
  ) -> node_collections.NodesCollection:
    """Constructs `NodeCollection`s for bonds between an annulus of radius 
     `nb_outer` and `nb_inner` nearest neighbour in the PXP rydberg Hamiltonian.
    
    Args:
      nb_outer: radius of outer annulus.
      nb_inner: radius of inner annulus.
    
    Returns:
      Bonds within an annulus of radius `nb_outer` and `nb_inner`. 
    """
    if self.boundary == 'open':
      nn_bonds = node_collections.get_nearest_neighbors(
          self._lattice, nb_outer, nb_inner
      )
    elif self.boundary == 'periodic':
      arity_fn = functools.partial(
          _distance_fn_mod, mod_vector=self.Ly * self.ruby_lattice.a2
      )
      nn_bonds = node_collections.get_nearest_neighbors(
          self._lattice, nb_outer, nb_inner, arity_fn
      )
    else:
      raise ValueError(
          f'`boundary` must be either `open` or `periodic`. '
          f'`{self.boundary=}` is not implemented.'
      )
    return nn_bonds
  
  def _get_nearest_neighbour_bonds(
      self,
  ) -> list[node_collections.NodesCollection]:
    """Constuct list of bonds for nearest neighbours between each annulus."""
    all_nn_bonds = []  # list of bonds between each annulus.
    for i in range(len(self.nb_radii)):
      if i == 0:  # first circle.
        nn_bonds = self._get_annulus_bonds(self.nb_radii[i])
      else:  # subsequent annulus.
        if self.nb_radii[i - 1] > self.nb_radii[i]:
          raise ValueError(
              f'`nb_radii` must be in ascending order. '
              f'{self.nb_radii[i - 1]=}` is greater than {self.nb_radii[i]=}`.'
          )        
        nn_bonds = self._get_annulus_bonds(
            self.nb_radii[i], self.nb_radii[i - 1]
        )
      all_nn_bonds.append(nn_bonds)
    return all_nn_bonds
  
  def _get_nearest_neighbour_groups(self) -> list[types.TermsTuple]:
    """Constuct terms for nearest neighbour bonds between each annulus."""
    all_nn_bonds = self._get_nearest_neighbour_bonds()
    all_nn_groups = []
    for i, bonds in enumerate(all_nn_bonds):
      terms = []
      for node in bonds.nodes:
        terms.append((self.vs[i] / 4., ('z', node[0]), ('z', node[1])))
        terms.append((+self.vs[i] / 4., ('z', node[0])))
        terms.append((+self.vs[i] / 4., ('z', node[1])))
        terms.append((+self.vs[i] / 4., ('I', node[0])))
      all_nn_groups.append(terms)
    return all_nn_groups

  def _get_onsite_groups(self) -> list[types.TermsTuple]:
    onsite_terms_z = []
    onsite_terms_x = []
    for i in range(self.n_sites):
      onsite_terms_z.append((-self.delta / 2., ('z', i)))
      onsite_terms_z.append((-self.delta / 2., ('I', i)))
    for i in range(self.n_sites):
      onsite_terms_x.append((self.omega / 2., ('x', i)))
    for i in self.boundary_sites:
      onsite_terms_z.append((-self.boundary_z_field / 2., ('z', i)))
      onsite_terms_z.append((-self.boundary_z_field / 2., ('I', i)))
    return [onsite_terms_z, onsite_terms_x]

  def _get_all_terms_groups(self) -> list[types.TermsTuple]:
    """Get all terms in hamiltonian as list of groups."""
    return self._get_nearest_neighbour_groups() + self._get_onsite_groups()
  
  def get_terms(self) -> types.TermsTuple:
    """Merge all terms from all groups into one tuple."""
    all_terms_groups = self._get_all_terms_groups()
    all_terms = []
    for group in all_terms_groups:
      all_terms += group
    return all_terms

  def get_ham(self) -> qtn.MatrixProductOperator:
    """Get hamiltonian as MPO."""
    # hamiltonian_mpo_groups = []
    # for terms in self._get_all_terms_groups():
    #   hamiltonian_mpo_groups.append(
    #       self.get_sparse_operator(terms).build_mpo()
    #   )  
    # #TODO(YT): consider building state machines for bulk 
    # and boundary separately.
    hamiltonian_mpos = []
    for term in self.get_terms():
      hamiltonian_mpos.append(
          self.get_sparse_operator([term]).build_mpo()
      )    
    ham_mpo = sum(hamiltonian_mpos[1:], start=hamiltonian_mpos[0])
    ham_mpo.compress()
    return ham_mpo


class RubyRydbergVanderwaals(RubyRydberg):
  """Implementation for Van der Waals ruby Rydberg interactions."""
  def __init__(
      self,
      Lx: int,
      Ly: int,
      delta: float = 5.0,
      boundary_z_field: float = 0.,
      rho: float = np.sqrt(3.),  
      rb: float = 3.8,  
      omega: float = 1.,
      nb_ratio_fn: Callable[[float], tuple[float, ...]] = lambda rho: (
          1., rho, np.sqrt(1. + rho**2)
      ),
      boundary: str = 'open',
  ):
    super().__init__(
        Lx=Lx, Ly=Ly, delta=delta, boundary_z_field=boundary_z_field, rho=rho,
        rb=rb, omega=omega, nb_ratio_fn=nb_ratio_fn, boundary=boundary
    )
    self._vs = np.array([(rb / r)**6 for r in nb_ratio_fn(self.rho)])


class RubyRydbergPXP(RubyRydberg):
  """Implementation for PXP ruby Rydberg interactions."""
  def __init__(
      self,
      Lx: int,
      Ly: int,
      delta: float = 5.0,
      boundary_z_field: float = 0.,
      rho: float = np.sqrt(3.),  
      rb: float = 3.8,  
      omega: float = 1.,
      nb_ratio_fn: Callable[[float], tuple[float, ...]] = lambda rho: (
          1., rho, np.sqrt(1. + rho**2)
      ),
      boundary: str = 'open',
  ):
    super().__init__(
        Lx=Lx, Ly=Ly, delta=delta, boundary_z_field=boundary_z_field, rho=rho,
        rb=rb, omega=omega, nb_ratio_fn=nb_ratio_fn, boundary=boundary
    )
    self._vs = (rb/(nb_ratio_fn(self.rho)[-1]))**6 * np.ones(len(self.nb_radii))
