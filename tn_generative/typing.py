from typing import Callable, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

Array = Union[np.ndarray, jnp.ndarray]
MeasurementAndBasis = Tuple[Array, Array]
SamplerFn = Callable[
    [jax.random.PRNGKeyArray, qtn.MatrixProductState], MeasurementAndBasis]
