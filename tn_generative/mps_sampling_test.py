"""Tests for mps_sampling."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
import quimb.gen as qugen

from tn_generative  import mps_sampling

# set backend to JAX for vmap and jit.
qtn.contraction.set_tensor_linop_backend('jax')
qtn.contraction.set_contract_backend('jax')

class MpsUtilsTests(parameterized.TestCase):
  """Tests for sampling matrix product state utilities."""

  @parameterized.parameters(2, 3, 4, 6, 9)
  def test_gibbs_sampler_neel(self, size, seed=42):
    """test neel state by sampling the shot configuration."""
    prodct_state = qtn.tensor_builder.MPS_neel_state(size, down_first=True)
    key = jax.random.PRNGKey(seed)
    actual_sample = mps_sampling.gibbs_sampler(key, prodct_state)
    expected_sample = ((1 + (-1) ** np.arange(size)) / 2).astype(int)
    np.testing.assert_array_equal(actual_sample, expected_sample)  

  @parameterized.parameters((2, 2), (3, 3),)
  def test_gibbs_sampler(self, size, bond_dim, seed=42):
    """Tests gibbs sampling method by estimating pdf of GHZ, random state."""
    def counts_to_probs(samples, bins):
      """Converts samples to probability distribution.
      #TODO (YT): move this to utils."""
      # Integer representation of binary strings for [N, L] array x. 
      # Convention: x[0] is the most significant bit.
      _binstr_to_int = lambda x: x.dot(1 << np.flip(np.arange(x.shape[-1])))  
      samples_arr_int = _binstr_to_int(samples)
      counts, _ = np.histogram(samples_arr_int, bins=bins)
      return counts / np.sum(counts)  

    # test GHZ state
    num_samples_ghz = 1000
    ghz_state = qtn.tensor_builder.MPS_ghz_state(size)
    key = jax.random.PRNGKey(seed)
    keys_sample = jax.random.split(key, num_samples_ghz)
    sample_fn = functools.partial(mps_sampling.gibbs_sampler, mps=ghz_state)
    gibbs_sampling_batched = jax.vmap(sample_fn)
    actual_samples = gibbs_sampling_batched(keys_sample)
    actual_pdf = counts_to_probs(actual_samples, bins=2 ** size)
    expected_pdf = jnp.squeeze(jnp.abs(ghz_state.to_dense()) ** 2)
    np.testing.assert_allclose(actual_pdf, expected_pdf, atol=1e-2)

    # test random state
    qugen.rand.seed_rand(seed)
    num_samples_random = 2000
    random_state = qtn.tensor_builder.MPS_rand_state(size, bond_dim=bond_dim)
    key = jax.random.PRNGKey(seed)
    keys_sample = jax.random.split(key, num_samples_random)
    sample_fn = functools.partial(mps_sampling.gibbs_sampler, mps=random_state)
    gibbs_sampling_batched = jax.vmap(sample_fn)
    actual_samples = gibbs_sampling_batched(keys_sample)
    bins = np.arange(2 ** size + 1) -0.5   # bin centers, starting from 0
    actual_pdf = counts_to_probs(actual_samples, bins=bins)
    expected_pdf = jnp.squeeze(jnp.abs(random_state.to_dense()) ** 2)
    np.testing.assert_allclose(
      abs(actual_pdf - expected_pdf), np.zeros_like(expected_pdf), atol=1e-2)


if __name__ == '__main__':
  absltest.main()