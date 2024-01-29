"""Physical observables."""
from typing import Sequence

import quimb as qu
import quimb.tensor as qtn
import quimb.experimental.operatorbuilder as quimb_exp_op

from tn_generative import physical_systems

def get_Q_mpo(
    three_sites: Sequence[int], 
    sparse_op: quimb_exp_op.SparseOperatorBuilder,
) -> qtn.MatrixProductOperator:
  """Construct Q operator for local Hilbert space of three sites.
  
  Q = - x1 n2 n3 + n1 (h2 + h3) x2 x3
  n = (1 + sz) / 2 is number operator
  h = (1 - sz) / 2 is hole operator
  Define new Q indices for the local triangle of sites (1, 2, 3)
  The order is such that the first site flips and 2 resonates with 3.
  See _get_Q_mpo for the definition of the Q operator
  and definition in Fig 5a of http://arxiv.org/abs/2011.12310 for the ordering
  of the sites.

  Args:
    three_sites: The three sites to construct the Q operator for.
    sparse_op: The sparse operator builder to use.
  
  Returns:
    The Q operator as an MPO.
  """
  site1, site2, site3 = three_sites
  sparse_op += -1.0, ('x', site1), ('n', site2), ('n', site3)
  sparse_op += 1.0, ('n', site1), ('h', site2), ('x', site2), ('x', site3)
  sparse_op += 1.0, ('n', site1), ('h', site3), ('x', site2), ('x', site3)
  return sparse_op.build_mpo()


# Motivation for why we need to get MPOs for each term in Q:
# We can't define terms for all Qs because each term is a sum of operators!!!
def compute_Q_order_params(
    mps: qtn.MatrixProductState,
    physical_system: physical_systems.PhysicalSystem, 
    q_bffm_indices: Sequence[Sequence[int]],
    q_loop_indices: Sequence[Sequence[int]],
) -> tuple[float]:
  """Compute order parameters for Q loop operators.

  Args:
    mps: The MPS to compute the order parameters for.
    physical_system: The physical system.
    q_bffm_indices: The indices of the Q operators for the BFFM.
    q_loop_indices: The indices of the Q operators for the loop.

  Returns:
    The order parameters for the Q operators: open and closed BFFM, and 
    the X logical operator.
  """
  sparse_operator = quimb_exp_op.SparseOperatorBuilder(
      hilbert_space=physical_system.hilbert_space
  )
  q_mpos = []
  for q in q_bffm_indices:
    if len(q) != 3:
      raise ValueError(
          f'Q_BFFM_indices must be a list of tuples of length 3, but got {q}')
    q_local_mpo = get_Q_mpo(q, sparse_operator)
    q_mpos.append(q_local_mpo)
  # compute open q loop order parameter
  mps_transformed = mps.copy()
  for q in q_mpos[:len(q_mpos) // 2]:
    mps_transformed = q.apply(mps_transformed)
    mps_transformed.compress() # necessary because of snake path ordering.
  mps_transformed.normalize()  
  q_open = mps.H @ mps_transformed
  # compute closed q loop order parameter
  for q in q_mpos[len(q_mpos) // 2:]:
    mps_transformed = q.apply(mps_transformed)
    mps_transformed.compress()
  mps_transformed.normalize()
  q_closed = mps.H @ mps_transformed
  # Compute loop order parameter (logical X)
  q_loop_mpos = []
  for q_loop in q_loop_indices:
    if len(q_loop) != 3:
      raise ValueError(f'Q_loop_indices must be a list of tuples of length 3, \
          but got {q_loop}.'
      )
    q_loop_local_mpo = get_Q_mpo(q_loop, sparse_operator)
    q_loop_mpos.append(q_loop_local_mpo)
  mps_transformed = mps.copy()
  for q_loop in q_loop_mpos: 
    mps_transformed = q_loop.apply(mps_transformed)
    mps_transformed.compress()
  mps_transformed.normalize()
  q_loop = mps.H @ mps_transformed
  return q_open, q_closed, q_loop


def compute_P_order_params(
    mps: qtn.MatrixProductState,
    physical_system: physical_systems.PhysicalSystem, 
    p_bffm_indices: Sequence[int],
    p_loop_indices: Sequence[int],
) -> tuple[float]:
  """Compute order parameters for P loop operators.
  
  Product of pauli Z operators for both open and closed BFFM, and logical. 
  See definition in Fig 5a of http://arxiv.org/abs/2011.12310 for the ordering
  of the sites.

  Args:
    mps: The MPS to compute the order parameters for.
    physical_system: The physical system.
    p_bffm_indices: The indices of the P operators for the BFFM.
    p_loop_indices: The indices of the P operators for the loop.
  
  Returns:
    The order parameters for the P operators: open and closed BFFM, and 
    the Z logical operator.
  """
  p_bffm_terms = [
    (1., *[('z', x) for x in p_bffm_indices]), 
    (1., *[('z', x) for x in p_bffm_indices[:len(p_bffm_indices) // 2]])
  ]
  p_bffm_ops = physical_system.get_obs_mpos(p_bffm_terms)
  p_closed, p_open = [mps.H @ op.apply(mps) for op in p_bffm_ops]
  # Compute loop order parameter (logical Z)
  p_loop_terms = [(1., *[('z', x) for x in p_loop_indices])]
  p_loop_operator = physical_system.get_obs_mpos(p_loop_terms)[0]
  p_loop = mps.H @ p_loop_operator.apply(mps)  
  return p_open, p_closed, p_loop
