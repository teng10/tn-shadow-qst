import jax
import jax.numpy as jnp
import numpy as np

def gibbs_sampler(key, mps):
  """Sample an observation from `mps` using gibbs sampling method.
  Args: 
    key: jax.random.PRNGKey.
    mps: qtn.MatrixProductState.
  Returns:
    outcomes (jnp.ndarray) of shape (L,) of integers in {0, 1} 
    where L is the system size.
  """
  mps = mps.copy()  # let's not modify outside copy here.
  keys = jax.random.split(key, mps.L)
  mps.canonize(0, cur_orthog=None)  # start with right canonical form.
  outcomes = []
  site_idx = 0  # we will iterate over 0th site as we project out sites.
  for i, rng in enumerate(keys):
    L = mps.L  # current length of the MPS.
    t = mps[site_idx]  # current tensor for sampling.
    site_ind = mps.site_ind(site_idx)  # name of the site index.
    # diagonal of reduced density matrix corresponds to measurement probs.
    t_ii = t.contract(t.H, output_inds=(site_ind,))
    probs = jnp.real(t_ii.data)
    # sample 0 or 1 depending on the likelihood.
    outcome = jax.random.choice(rng, np.arange(mps.phys_dim()), [], p=probs)
    # project the outcome of the measurement.
    t.isel_({site_ind: outcome})
    # renormalize.
    t.modify(data=t.data / probs[outcome]**0.5)
    # contract projected tensor into the MPS and retag/reindex.
    if site_idx == L - 1:
      mps ^= slice(site_idx - 1, site_idx + 1)
    else:
      mps ^= slice(site_idx, site_idx + 2)
    for i in range(site_idx + 1, L):
      mps[i].reindex_({mps.site_ind(i): mps.site_ind(i - 1)})
      mps[i].retag_({mps.site_tag(i): mps.site_tag(i - 1)})
    mps._L = L - 1
    outcomes.append(outcome)
  return jnp.stack(outcomes)  