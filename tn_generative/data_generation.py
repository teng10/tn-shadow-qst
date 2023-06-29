#@title Imports

from __future__ import annotations

import functools
import itertools
import operator
import tqdm

import dataclasses
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import einops
import haiku as hk
import optax

import pandas as pd
import xarray as xr
import xyzpy

# import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

# Quimb imports for tensor network manipulations
import quimb as qmb
import quimb.tensor as qmbt
from quimb.experimental import operatorbuilder as quimb_exp_op

# Used for lattice building utilities.
import scipy.spatial as sp_spatial
import shapely
from shapely.geometry.polygon import Polygon

from ml_collections import config_dict


Array = Union[np.ndarray, jnp.ndarray]
shape_structure = lambda tree: jax.tree_util.tree_map(lambda x: x.shape, tree)

