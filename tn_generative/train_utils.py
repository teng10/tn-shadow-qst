"""Training and evaluation helper functions."""
import functools
from typing import Any, Dict, Sequence, Tuple
import time
import tqdm

import pandas as pd
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import optax
import quimb.tensor as qtn
import tensorflow as tf


from tn_generative import data_generation
from tn_generative import data_utils
from tn_generative import mps_utils
from tn_generative import func_utils
from tn_generative import regularizers


TRAIN_SCHEME_REGISTRY = {}  # define registry for training scheme step function.

register_training_scheme = func_utils.get_register_decorator(
    TRAIN_SCHEME_REGISTRY
)

REGULARIZER_REGISTRY = regularizers.REGULARIZER_REGISTRY
TASK_REGISTRY = data_generation.TASK_REGISTRY


def measurement_log_likelihood(
    mps: qtn.MatrixProductState,
    measurement: jax.Array,
    basis: jax.Array | None = None,
) -> float:
  """Evaluates log-likelihood of measurement for MPS described by `mps`.
  """
  amplitude = mps_utils.amplitude_via_contraction(mps, measurement, basis)
  sqrt_probability = jnp.abs(amplitude)
  return 2 * (jnp.log(sqrt_probability))


@jax.jit
def batched_neg_ll_loss_fn(
    mps_arrays: Sequence[jax.Array],
    measurements: jax.Array,
    bases: jax.Array | None = None,
) -> float:
  """Batched negative log likelihood loss function."""
  batched_ll_fn = jax.vmap(measurement_log_likelihood, in_axes=(None, 0, 0))
  mps = qtn.MatrixProductState(arrays=mps_arrays)
  norm = mps.norm().real  # NOTE: can compute norm once and reuse.
  loss_fn = lambda p, m, b: (-jnp.mean(batched_ll_fn(p, m, b))
    + 2 * jnp.log(norm)
  )
  return loss_fn(mps, measurements, bases)


def evaluate_model(mps, train_ds, test_ds, regularization_fn):
  """Evaluates `mps` wrt learning the state in `train_ds`.

  This function computes: (1) fieldlity - how well the ground truth state is
  learned from data; (2) log likelihood of data under model and ground truth
  state per data sample (i.e. mean of log likelihood).
  NOTE: currenty log-likelihood is evaluated in a single pass, which might run
  out of memory if the dataset is very large. This could be fixed by computing
  the mean in a streaming fashion.
  #TODO(YT): implement streaming mean for log-likelihood computation.

  Args:
    mps: trained state to evaluate.
    train_ds: default tomography dataset containing `measurement`, `basis` vars
      on which `mps` was trained on, as well as parameters of the ground truth
      state which are used to reconstruct the state and compute fidelity.
    test_ds: test dataset to evaluate the model on.  
    regularization_fn: function that computes regularization term.

  Returns:
    Dictionary containing evaluation summary: fildelity, model_ll, target_ll,
    regularization (if not None) and test_ll.
  """
  target_mps = mps_utils.xarray_to_mps(train_ds)
  fidelity = (mps.H | target_mps) ^ ...
  # Evaluate log-likelihood on training dataset.
  measurements = train_ds.measurement.values
  bases = train_ds.basis.values
  model_ll = batched_neg_ll_loss_fn(mps.arrays, measurements, bases)
  target_ll = batched_neg_ll_loss_fn(target_mps.arrays, measurements, bases)
  # Evaluate on test dataset.
  test_measurements = test_ds.measurement.values
  test_bases = test_ds.basis.values
  test_ll = batched_neg_ll_loss_fn(mps.arrays, test_measurements, test_bases)  
  if regularization_fn is not None:
    regularization = regularization_fn(mps.arrays)
  else:
    regularization = np.nan
  return pd.DataFrame({
      'fidelity': [np.abs(fidelity)],
      'model_ll': [model_ll],
      'target_ll': [target_ll],
      'regularization': [regularization],
      'test_ll': [test_ll],
  })


@register_training_scheme('lbfgs')
def run_full_batch_training(
    mps: qtn.MatrixProductState,
    train_ds: xr.Dataset,
    test_ds: xr.Dataset,
    training_config: Dict[str, Any],
    num_training_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, qtn.MatrixProductState]:
  """Runs training with full-batch optimization using LBFGS.

  Args:
    mps: initial state to train.
    train_ds: default tomography dataset containing `measurement`, `basis`.
    test_ds: test dataset to evaluate the model on.
    training_config: dictionary containing training configuration.
    num_training_steps: number of training steps.

  Returns:
    train_df: pandas dataframe containing training loss and optimization step.
    eval_df: pandas dataframe containing evaluation metrics.
    trained_mps: trained MPS state.
  """
  measurements = train_ds.measurement.values
  bases = train_ds.basis.values
  physical_system = data_utils.physical_system_from_attrs_dict(train_ds.attrs)
  get_regularization_fn = REGULARIZER_REGISTRY[training_config.reg_name]
  if get_regularization_fn is not None:
    regularization_fn = get_regularization_fn(
        system=physical_system, train_ds=train_ds,
        **training_config.reg_kwargs
    )
    loss_fn = lambda psi, m, b: (batched_neg_ll_loss_fn(psi.arrays, m, b)
    + regularization_fn(psi.arrays))  #TODO(YT): is there a difference to jit?
  else:
    loss_fn = lambda psi, m, b: batched_neg_ll_loss_fn(psi.arrays, m, b)
    regularization_fn = None  # for eval_df, otherwise raise an error.
  loss_fn = functools.partial(loss_fn, m=measurements, b=bases)
  tnopt = qtn.TNOptimizer(
      mps,
      loss_fn=loss_fn,
      norm_fn=mps_utils.uniform_normalize,  # use normalize that acts on `mps`.
      autodiff_backend='jax',
  )
  start_training_time = time.time()
  trained_mps = tnopt.optimize(num_training_steps)
  end_training_time = time.time()
  train_df = pd.DataFrame({
      'loss': tnopt.losses,
      'opt_step': np.arange(len(tnopt.losses)),
  })
  train_df['training_time'] = end_training_time - start_training_time
  train_df = train_df.astype(np.float32)
  eval_df = evaluate_model(trained_mps, train_ds, test_ds, regularization_fn)
  return train_df, eval_df, trained_mps


@register_training_scheme('minibatch')
def run_minibatch_trainig(
    mps: qtn.MatrixProductState,
    train_ds: xr.Dataset,
    test_ds: xr.Dataset,
    training_config: Dict[str, Any],
    num_training_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, qtn.MatrixProductState]:
  """Runs training using adam on mini-batches of data.

  Args:
    mps: initial state to train.
    train_ds: default tomography dataset containing `measurement`, `basis`.
    test_ds: test dataset to evaluate the model on.
    training_config: dictionary containing training configuration.
    num_training_steps: number of training steps.

  Returns:
    train_df: pandas dataframe containing training loss and optimization step.
    eval_df: pandas dataframe containing evaluation metrics.
    trained_mps: trained MPS state.
  """
  physical_system = data_utils.physical_system_from_attrs_dict(train_ds.attrs)
  get_regularization_fn = REGULARIZER_REGISTRY[training_config.reg_name]
  if get_regularization_fn is not None:
    regularization_fn = get_regularization_fn(
        system=physical_system, train_ds=train_ds,
        **training_config.reg_kwargs
    )
    loss_fn = lambda psi_arrays, m, b: (batched_neg_ll_loss_fn(psi_arrays, m, b)
    + regularization_fn(psi_arrays))  #TODO(YT): difference to jit? -> No
  else:
    loss_fn = lambda psi_arrays, m, b: batched_neg_ll_loss_fn(psi_arrays, m, b)
    regularization_fn = None  # for eval_df, otherwise raise an error.

  loss_and_grad_fn = jax.value_and_grad(jax.jit(loss_fn))

  params = mps.copy().arrays
  #TODO(YT): try different optimizers?
  opt_init_fn, opt_update_fn = optax.adam(**training_config.opt_kwargs)
  opt_state = opt_init_fn(params)

  @jax.jit
  def train_step_fn(params, opt_state, measurement, basis):
    loss_value, grad = loss_and_grad_fn(params, measurement, basis)
    grad = jax.tree_util.tree_map(jnp.conjugate, grad)
    updates, opt_state = opt_update_fn(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = mps_utils.uniform_param_normalize(params)
    # grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grad)
    return params, opt_state, loss_value

  training_kwargs = training_config.training_kwargs
  tf_dataset = tf.data.Dataset.from_tensor_slices(
      (train_ds.measurement.values, train_ds.basis.values))
  tf_dataset = tf_dataset.shuffle(buffer_size=1024 * 20)
  tf_dataset = tf_dataset.cache()
  tf_dataset = tf_dataset.batch(training_kwargs['batch_size'])
  tf_dataset = tf_dataset.repeat()  # makes it infinite.
  train_iter = tf_dataset.as_numpy_iterator()

  pbar = tqdm.tqdm(range(num_training_steps),
                   desc='Step', position=0, leave=True)
  results = []
  start_training_time = time.time()
  for i in pbar:
    measurement, basis = next(train_iter)
    params, opt_state, l = train_step_fn(params, opt_state, measurement, basis)
    if i % training_kwargs['record_loss_interval'] == 0:
      metrics_dict = {
          'loss': jax.device_get(l),
          'opt_step': jax.device_get(i),
      }
      results.append(metrics_dict)
      pbar.set_postfix(results[-1])
  end_training_time = time.time()
  trained_mps = qtn.MatrixProductState(params)
  train_df = pd.DataFrame(jax.device_get(results))
  train_df['training_time'] = end_training_time - start_training_time
  train_df = train_df.astype(np.float32)
  eval_df = evaluate_model(trained_mps, train_ds, test_ds, regularization_fn)
  return train_df, eval_df, trained_mps
