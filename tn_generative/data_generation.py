"""Data generation for training MPS models."""
import functools

from tn_generative import physical_systems

PhysicalSystem = physical_systems.PhysicalSystem

TASK_REGISTRY = {}  # define registry for task hamiltonians.


def _register_task(get_task_system_fn, task_name: str):
  """Registers `get_task_system_fn` in global `TASK_REGISTRY`."""
  registered_fn = TASK_REGISTRY.get(task_name, None)
  if registered_fn is None:
    TASK_REGISTRY[task_name] = get_task_system_fn
  else:
    if registered_fn != get_task_system_fn:
      raise ValueError(f'{task_name} is already registerd {registered_fn}.')


register_task = lambda name: functools.partial(_register_task, task_name=name)


@register_task('surface_code')
def get_surface_code(
    size_x: int,
    size_y: int,
    coupling_value: float = 1.0,
    onsite_z_field: float = 0.,
) -> PhysicalSystem:
  """Generates surface code for `[size_x, size_y]` domain with specification
  of stabilizer coupling `coupling_value` and external field `onsite_z_field`.
  """
  return physical_systems.SurfaceCode(
      size_x, size_y, coupling_value, onsite_z_field
  )


@register_task('ruby_vanderwaals')
def get_ruby_vanderwaals(
    size_x: int,
    size_y: int,
    delta: float = 5.,
) -> PhysicalSystem:
  """Generates ruby rydberg physical system for `[size_x, size_y]` domain
  with specification of detuning `delta` parameter."""
  #TODO(YT): remove this or complete docstring.
  return physical_systems.RubyRydbergVanderwaals(size_x, size_y, delta)


@register_task('ruby_pxp')
def get_ruby_pxp(
    size_x: int,
    size_y: int,
    delta: float = 5.,
    boundary_z_field: float = 0.,
    boundary: str = 'open',
) -> PhysicalSystem:
  """Generates ruby PXP physical system for `[size_x, size_y]` domain
  with specification of detuning `delta` parameter with `boundary`.

  Args:
    size_x: system size in x direction.
    size_y: system size in y direction.
    delta: detuning parameter.
    boundary_z_field: boundary z field for boundary corrections.
    boundary: boundary condition: `periodic` or `open`.
  Returns:
    physical system.
  """
  # COMMENT: `boundary_z_field` is a hack to offset boundary effects.
  # Rather than only adding delta, try mean-field balue
  # boundary_z_field= 3 * <n> ~ -0.6  
  return physical_systems.RubyRydbergPXP(
      size_x, size_y, delta, boundary=boundary,
      boundary_z_field=boundary_z_field
  )
