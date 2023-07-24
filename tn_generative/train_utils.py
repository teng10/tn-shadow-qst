"""Training and evaluation helper functions."""
from typing import Any, Dict, Optional, Sequence
import functools

import pandas as pd
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

from tn_generative import mps_utils

def measurement_log_likelihood(
    mps: qtn.MatrixProductState, 
    measurement: jax.Array,
    basis: jax.Array | None = None,
) -> float:
  """Evaluates log-likelihood of measurement for MPS described by `mps`.
  """
  # mps = qtn.MatrixProductState(arrays=mps_arrays)  #TODO: remove this line.
  amplitude = mps_utils.amplitude_via_contraction(mps, measurement, basis)
  sqrt_probability = jnp.abs(amplitude)
  return 2 * (jnp.log(sqrt_probability))


@jax.jit
def batched_neg_ll_loss_fn(
    mps_arrays: Sequence[jax.Array],
    measurements: jax.Array,
    bases: jax.Array | None = None,  # Question: why is `| None` here?
):
  """Batched negative log likelihood loss function."""
  batched_ll_fn = jax.vmap(measurement_log_likelihood, in_axes=(None, 0, 0))
  mps = qtn.MatrixProductState(arrays=mps_arrays)
  norm = mps.norm().real  # NOTE: can compute norm once and reuse.
  loss_fn = lambda p, m, b: (-jnp.mean(batched_ll_fn(p, m, b)) 
    + 2 * jnp.log(norm)
  )
  return loss_fn(mps, measurements, bases)


def evaluate_model(mps, train_ds):
  """Evaluates `mps` wrt learning the state in `train_ds`.

  This function computes: (1) fieldlity - how well the ground truth state is
  learned from data; (2) log likelihood of data under model and ground truth
  state per data sample (i.e. mean of log likelihood).
  NOTE: currenty log-likelihood is evaluated in a single pass, which might run
  out of memory if the dataset is very large. This could be fixed by computing
  the mean in a streaming fashion.

  Args:
    mps: trained state to evaluate.
    train_ds: default tomography dataset containing `measurement`, `basis` vars
      on which `mps` was trained on, as well as parameters of the ground truth
      state which are used to reconstruct the state and compute fidelity.

  Returns:
    Dictionary containing evaluation summary: fildelity, model_ll, target_ll.
  """
  target_mps = mps_utils.xarray_to_mps(train_ds)
  fidelity = (mps.H | target_mps) ^ ...

  measurements = train_ds.measurement.values
  bases = train_ds.basis.values
  model_ll = batched_neg_ll_loss_fn(mps.arrays, measurements, bases)
  target_ll = batched_neg_ll_loss_fn(target_mps.arrays, measurements, bases)
  return pd.DataFrame({
      'fidelity': [np.abs(fidelity)],
      'model_ll': [model_ll],
      'target_ll': [target_ll],
  })


#@title Defining two training variants: (1) full batch LBFGS; (2) mini-batch SGD
def run_full_batch_training_regularizer(
    mps: qtn.MatrixProductState,
    train_ds: xr.Dataset,
    training_config: Dict[str, Any], 
    regularization_fn: Optional[callable]=None,
    beta: Optional[float]=1.
):
  """Runs training with full-batch optimization using LBFGS.

  Args:
    mps: initial state to train.
    train_ds: default tomography dataset containing `measurement`, `basis`.
    training_config: dictionary containing training configuration.
    regularization_fn: function that computes regularization term.
    beta: regularization strength.
  
  Returns:
    train_df: pandas dataframe containing training loss and optimization step.
  """
  measurements = train_ds.measurement.values
  bases = train_ds.basis.values

  regularization_fn = functools.partial(regularization_fn, train_ds=train_ds)
  if regularization_fn is not None:
    loss_fn = lambda psi, m, b: (batched_neg_ll_loss_fn(psi.arrays, m, b) 
    + beta * regularization_fn(psi.arrays))
  else:
    loss_fn = lambda psi, m, b: batched_neg_ll_loss_fn(psi.arrays, m, b)
  tnopt = qtn.TNOptimizer(
      mps,
      loss_fn=jax.jit(loss_fn),
      norm_fn=mps_utils.uniform_normalize,  # use normalize that acts on `mps`.
      loss_constants={"m": measurements, "b": bases},
  )
  trained_mps = tnopt.optimize(training_config.num_training_steps)
  train_df = pd.DataFrame({
      'loss': tnopt.losses,
      'opt_step': np.arange(len(tnopt.losses)),
  })
  train_df = train_df.astype(np.float32)
  eval_df = evaluate_model(trained_mps, train_ds)
  return train_df, eval_df, trained_mps
