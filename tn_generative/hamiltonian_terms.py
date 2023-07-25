#@title || Surface code components implementations


def get_surface_code_collections(size_x, size_y):
  """Constructs `NodeCollection`s for all terms in surface code Hamiltonian."""
  if size_x % 2 != 1 or size_y % 2 !=1:
    raise NotImplementedError('Only odd system sizes are implemented.')
  a1 = np.array([1.0, 0.0])  # unit vectors for square lattice.
  a2 = np.array([0.0, 1.0])
  unit_cell_points = np.stack([np.array([0, 0])])
  unit_cell = Lattice(unit_cell_points)

  lattice = sum(
      [unit_cell.shift(a1 * i + a2 * j)
      for i, j in itertools.product(range(size_x), range(size_y))]
  )
  # generating terms with Z plaquetts at the top left corner.
  z_top_loc = np.array((0, size_y - 1))
  z_base_plaquette = build_path_from_sequence(
      lattice.get_idx(
          np.stack(  # specify plaquet going right, down, right and back up.
              [z_top_loc, z_top_loc + a1, z_top_loc + a1 - a2, z_top_loc - a2])
          ),
      lattice
  )
  # fixing boundary terms with x-x interactions for Z plaquetts on the boundary.
  x_top_boundary = build_path_from_sequence(
      lattice.get_idx(np.stack([z_top_loc, z_top_loc + a1])), lattice
  )
  x_bot_boundary = x_top_boundary.shift(-a2 * (size_y - 1) + a1)

  # generating terms with X plaquetts at the second square from top left corner.
  x_top_loc = z_top_loc + a1
  x_base_plaquette = build_path_from_sequence(
      lattice.get_idx(
          np.stack(
              [x_top_loc, x_top_loc + a1, x_top_loc + a1 - a2, x_top_loc - a2])
          ),
      lattice
  )
  # fixing boundary terms with z-z interactions for X plaquetts on the boundary.
  z_left_boundary = build_path_from_sequence(
      lattice.get_idx(np.stack([z_top_loc - a2, z_top_loc - 2 * a2])), lattice
  )
  z_right_boundary = z_left_boundary.shift(a1 * (size_x - 1) + a2)

  n_shifts = max(size_x, size_y)
  z_plaquettes = tile_on_lattice(z_base_plaquette, 2 * a1, a1 + a2, n_shifts)
  x_plaquettes = tile_on_lattice(x_base_plaquette, 2 * a1, a1 + a2, n_shifts)
  x_boundaries = (
      tile_on_lattice(x_top_boundary, 2 * a1, 0 * a2, n_shifts) +
      tile_on_lattice(x_bot_boundary, 2 * a1, 0 * a2, n_shifts))
  z_boundaries = (
      tile_on_lattice(z_left_boundary, 0 * a1, 2 * a2, n_shifts) +
      tile_on_lattice(z_right_boundary, 0 * a1, 2 * a2, n_shifts))
  return z_plaquettes, x_plaquettes, x_boundaries, z_boundaries


def surface_code_from_plaquettes(
    z_plaquettes: list[int, int, int, int],
    x_plaquettes: list[int, int, int, int],
    x_boundaries: list[int, int],
    z_boundaries: list[int, int],
    coupling_value: float = 1.0,
    hilbert_space: Optional[quimb_exp_op.HilbertSpace] = None,
) -> quimb_exp_op.SparseOperatorBuilder:
  """Generates surface code H from 4-plaquettes and 2-site boundaries.

  Note: this constructor assumes default ferromagnetic coupling. i.e.
  `coupling_value == 1` corresponds to `H = -1 * (Σ A) ...`.

  Args:
    z_plaquettes: collection of 4-sites forming σz plaquettes.
    x_plaquettes: collection of 4-sites forming σx plaquettes.
    x_boundaries: collection of 2 boundary sites on Z plaquettes with x-x term.
    z_boundaries: collection of 2 boundary sites on X vertices with z-z term.
    coupling_value: coupling coefficient.
    hilbert_space: optional HilertSpace instance.

  Returns:
    Sparse operator class representing the surface code hamiltonian.
  """
  H = quimb_exp_op.SparseOperatorBuilder(hilbert_space=hilbert_space)
  for i, j, k, l in z_plaquettes:
    H += -coupling_value, ("z", i), ("z", j), ("z", k), ("z", l)
  for i, j, k, l in x_plaquettes:
    H += -coupling_value, ("x", i), ("x", j), ("x", k), ("x", l)

  for i, j in x_boundaries:
    H += -coupling_value, ("x", i), ("x", j)
  for i, j in z_boundaries:
    H += -coupling_value, ("z", i), ("z", j)
  return H


def surface_code_hamiltonian(
    size_x: int,
    size_y: int,
    coupling_value: float = 1.0,
) -> quimb_exp_op.SparseOperatorBuilder:
  """Generates surface code hamiltonian for `[size_x, size_y]` domain."""
  z_plaquettes, x_plaquettes, x_boundaries, z_boundaries = (
      get_surface_code_collections(size_x, size_y))

  # extract edges specified by lattice sites for each interaction term.
  z_plaquett_edges = extract_edge_sequence(z_plaquettes)
  x_plaquett_edges = extract_edge_sequence(x_plaquettes)
  xx_boundary_edges = extract_edge_sequence(x_boundaries)
  zz_boundary_edges = extract_edge_sequence(z_boundaries)
  # get sites for each interaction term by selecting sites.
  z_plaquett_sites = np.unique(
      einops.rearrange(z_plaquett_edges, 'b i s -> b (i s)'), axis=-1)
  x_plaquett_sites = np.unique(
      einops.rearrange(x_plaquett_edges, 'b i s -> b (i s)'), axis=-1)
  z_boundary_sites = np.unique(
      einops.rearrange(zz_boundary_edges, 'b i s -> b (i s)'), axis=-1)
  x_boundary_sites = np.unique(
      einops.rearrange(xx_boundary_edges, 'b i s -> b (i s)'), axis=-1)
  return surface_code_from_plaquettes(
      z_plaquettes=z_plaquett_sites,
      x_plaquettes=x_plaquett_sites,
      x_boundaries=x_boundary_sites,
      z_boundaries=z_boundary_sites,
      coupling_value=coupling_value)