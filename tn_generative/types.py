"""Typing definitions for the tn_generative package."""
from typing import Callable, Union

import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
import quimb.experimental.operatorbuilder as quimb_exp_op

Array = Union[np.ndarray, jnp.ndarray]
MeasurementAndBasis = tuple[Array, Array]
TermsTuple = list[tuple[float, tuple[str, int]]]
HilbertSpace = quimb_exp_op.HilbertSpace
SamplerFn = Callable[
    [jax.random.PRNGKeyArray, qtn.MatrixProductState], MeasurementAndBasis]
DTYPES_REGISTRY = {
    'complex128': np.complex128,
    'complex64': np.complex64,
    'float64': np.float64,
    'float32': np.float32,
}
