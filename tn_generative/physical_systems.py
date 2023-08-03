"""Physical systems for generating dataset."""
import abc
from abc import abstractmethod
import itertools
import functools
from typing import Optional

import numpy as np
import einops
import quimb.tensor as qtn
import quimb.experimental.operatorbuilder as quimb_exp_op

from tn_generative import node_collections
from tn_generative import lattices
from tn_generative import types


class PhysicalSystem(abc.ABC):
  """Abstract class for defining physical systems."""

  @abstractmethod
  def get_terms(self) -> types.TermsTuple:
    """Returns list of terms in the hamiltonian."""
    return None

  @abstractmethod
  def get_ham(self) -> qtn.MatrixProductOperator:
    """Returns a hamiltonian MPO."""

  @abstractmethod
  def get_ham_mpos(self,
  ) -> list[qtn.MatrixProductOperator]:
    """Returns MPOs for list of terms in the hamiltonian.

    Note: this method returns terms in `get_terms` method with +1 coupling.

    Return:
      List of MPOs in the hamiltonian.
    """
    if self.get_terms() is None:
      raise ValueError(
          f'subclass {self.__name__} did not implement custom `get_terms`'
      )

  @abstractmethod
  def get_obs_mpos(self,
      terms: Optional[types.TermsTuple],
  ) -> list[qtn.MatrixProductOperator]:
    """Returns MPOs for observables.

    Args:
      terms: list of terms to include in the MPOs.

    Return:
      List of MPOs.
    """

class SurfaceCode(PhysicalSystem):
  """Implementation for surface code.
    Note: this constructor assumes default ferromagnetic coupling. i.e.
    `coupling_value == 1` corresponds to `H = -1 * (Σ A) ...`.
    Note: `get_terms` method in this constructor assumes +1 coupling.
  """

  def __init__(self,
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
    self.hilbert_space = quimb_exp_op.HilbertSpace(self.n_sites)

  def _get_surface_code_collections(self,
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

  def get_terms(self,
  ) -> types.TermsTuple:
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

  def _get_surface_code_hamiltonian_builder(self
  ) -> quimb_exp_op.SparseOperatorBuilder:
    """Generates surface code hamiltonian operator builder."""
    surface_code_ham = quimb_exp_op.SparseOperatorBuilder(
        hilbert_space=self.hilbert_space
    )
    for term in self.get_terms():  # add all stabilizers to the hamiltonian.
      surface_code_ham += term
    return surface_code_ham

  def get_ham(self,
  ) -> qtn.MatrixProductOperator:
    """Get surface code hamiltonian as MPO."""
    surface_code_ham = self._get_surface_code_hamiltonian_builder()
    return surface_code_ham.build_mpo()

  def get_ham_mpos(self,
  ) ->  list[qtn.MatrixProductOperator]:
    """Get observables in `get_terms` as MPOs.

    Returns:
      List of MPOs in the hamiltonian.
    """
    hilbert_space = self.hilbert_space
    mpos = []
    terms = [(1., *term[1:]) for term in self.get_terms()]
    for term in terms:
      mpos.append(
          quimb_exp_op.SparseOperatorBuilder(
              [term], hilbert_space=hilbert_space).build_mpo()
      )
    return mpos

  def get_obs_mpos(self,
      terms: Optional[types.TermsTuple] = None,
  ) ->  list[qtn.MatrixProductOperator]:
    """Get observables `terms` as MPOs.

    Args:
      terms: list of terms to include in the MPOs. Default (None) is to include
        all terms in the surface code hamiltonian.

    Returns:
      List of MPOs.
    """
    hilbert_space = self.hilbert_space
    mpos = []
    for term in terms:
      mpos.append(
          quimb_exp_op.SparseOperatorBuilder(
              [term], hilbert_space=hilbert_space).build_mpo()
      )
    return mpos


class RubyRydbergArraysPXP(PhysicalSystem):
  """Implementation for ruby Rydberg hamiltonian.
    Note: this constructor assumes PXP model uniform coupling `v` among three 
    nearest neibours. The range of neibours are determined by `nb_radii`, 
    specified in ascending order.

  """

  def __init__(self,
      Lx: int,
      Ly: int,
      delta: float = 5.0,
      v: float = 50.,
      omega: float = 1.,
      nb_radii: tuple[float] = (0.25, 2. * np.sqrt(3) / 8. , 0.5),
  ):
    self.n_sites = int(Lx * Ly * 6)
    self.Lx = Lx
    self.Ly = Ly
    self.delta = delta
    self.v = v
    self.omega = omega
    self.epsilon = 1e-3
    self.nb_radii = tuple(r + self.epsilon for r in nb_radii)
    self.hilbert_space = quimb_exp_op.HilbertSpace(self.n_sites)

  def _get_annulus_bonds(self,
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
    unit_cell_points = np.array(
        [[1. / 4., 0.], [1./ 8., np.sqrt(3) / 8.], 
        [3. / 8., np.sqrt(3) / 8.], [1. / 8., 3. * np.sqrt(3) / 8.],
        [3. / 8., 3. * np.sqrt(3) / 8.], [1. / 4., np.sqrt(3) / 2.]]
    )
    unit_cell = lattices.Lattice(unit_cell_points)

    a1 = np.array([1.0, 0.0])  # unit vectors for square lattice.
    a2 = np.array([0.5, np.sqrt(3.0)/2])
    expanded_lattice = sum(
        unit_cell.shift(a1 * i + a2 * j)
        for i, j in itertools.product(range(self.Lx), range(self.Ly))
    )

    nn_bonds = node_collections.get_nearest_neighbors(
        expanded_lattice, nb_outer, nb_inner
    )
    return nn_bonds
  
  def _get_nearest_neighbour_bonds(self,
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
  
  def _get_nearest_neighbour_groups(self,
  ) -> list[types.TermsTuple]:
    """Constuct terms for nearest neighbour bonds between each annulus."""
    all_nn_bonds = self._get_nearest_neighbour_bonds()
    all_nn_groups = []
    for bonds in all_nn_bonds:
      terms = []
      for node in bonds.nodes:
        terms.append((self.v, ('z', node[0]), ('z', node[1])))
      all_nn_groups.append(terms)
    return all_nn_groups

  def _get_onsite_groups(self,
  ) -> list[types.TermsTuple]:
    onsite_terms_z = []
    onsite_terms_x = []
    for i in range(self.n_sites):
      onsite_terms_z.append((-self.delta, ('z', i)))
    for i in range(self.n_sites):
      onsite_terms_x.append((self.omega / 2., ('x', i)))
    return [onsite_terms_z, onsite_terms_x]

  def _get_all_terms_groups(self,
  ) -> list[types.TermsTuple]:
    """Get all terms in hamiltonian as list of groups."""
    return self._get_nearest_neighbour_groups() + self._get_onsite_groups()
  
  def get_terms(self
  ) -> types.TermsTuple:
    """Merge all terms from all groups into one tuple."""
    all_terms_groups = self._get_all_terms_groups()
    all_terms = []
    for group in all_terms_groups:
      all_terms += group
    return all_terms

  def _get_hamiltonian_builder(self, 
  terms: types.TermsTuple,
  ) -> quimb_exp_op.SparseOperatorBuilder:
    """Generates hamiltonian including `terms` using sparse operator builder."""
    sparse_hamiltonian = quimb_exp_op.SparseOperatorBuilder(
        hilbert_space=self.hilbert_space
    )
    for term in terms:  # add all terms to the hamiltonian.
      sparse_hamiltonian += term
    return sparse_hamiltonian

  def get_ham(self,
  ) -> qtn.MatrixProductOperator:
    """Get hamiltonian as MPO."""
    hamiltonian_mpo_groups = []
    for terms in self._get_all_terms_groups():
      hamiltonian_mpo_groups.append(
          self._get_hamiltonian_builder(terms).build_mpo()
      )
    ham = hamiltonian_mpo_groups[0]
    # Question: sum(hamiltonian_mpo_groups) does not work?
    return functools.reduce(lambda x, y: x + y, hamiltonian_mpo_groups[1:], ham)

  def get_ham_mpos(self,
  ) ->  list[qtn.MatrixProductOperator]:
    """Get observables in `get_terms` as MPOs.

    Returns:
      List of MPOs in the hamiltonian.
    """
    hilbert_space = self.hilbert_space
    mpos = []
    terms = [(1., *term[1:]) for term in self.get_terms()]
    for term in terms:
      mpos.append(
          quimb_exp_op.SparseOperatorBuilder(
              [term], hilbert_space=hilbert_space).build_mpo()
      )
    return mpos

  def get_obs_mpos(self,
      terms: Optional[types.TermsTuple] = None,
  ) ->  list[qtn.MatrixProductOperator]:
    """Get observables `terms` as MPOs.

    Args:
      terms: list of terms to include in the MPOs. Default (None) is to include
        all terms in the hamiltonian.

    Returns:
      List of MPOs.
    """
    hilbert_space = self.hilbert_space
    mpos = []
    for term in terms:
      mpos.append(
          quimb_exp_op.SparseOperatorBuilder(
              [term], hilbert_space=hilbert_space).build_mpo()
      )
    return mpos
