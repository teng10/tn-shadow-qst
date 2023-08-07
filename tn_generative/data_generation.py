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


@register_task('cluster_state')
def get_cluster_state(
    size_x: int,
    size_y: int,
    onsite_z_field: float = 0.0,
) -> PhysicalSystem:
  """Generates cluster state for `[size_x, size_y]` domain with specification
  of stabilizer coupling `coupling_value`.
  """
  return physical_systems.ClusterState(size_x, size_y, onsite_z_field)
