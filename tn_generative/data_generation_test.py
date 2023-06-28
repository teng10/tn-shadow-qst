# #@title || test for mpo/mps contraction (mps_utils_test.py)

# def test_mps_mpo_contraction():
#   mps = qmbt.MPS_rand_state(3, bond_dim=5)
#   mpo = qmbt.MPO_product_operator([Y_HADAMARD, Y_HADAMARD, EYE])
#   mpo2 = qmbt.MPO_product_operator([Y_HADAMARD, Y_HADAMARD, EYE], upper_ind_id='b{}', lower_ind_id='k{}') # lower one needs to match the mps index
#   # Note: if we contract the mps and mpo tensors by hand, need to flip the indices!
#   # see quimb doc on `tensor_network_apply_op_vec`: 
#   # "The returned tensor network has the same site indices as ``tn_vec``, and it is the ``lower_ind_id`` of ``tn_op`` that is contracted."
#   mps1 = mpo.apply(mps) # k
#   mps2 = (mpo2 | mps) ^ ... # b--> k'

#   basis = np.array([1, 2, 3, 3, 0])
#   arrays = jax.nn.one_hot(basis[:3] % 2, mps1.phys_dim())
#   arrays = [x[0] for x in jnp.split(arrays, mps1.L)]
#   bit_state1 = qmbt.MPS_product_state(arrays, site_ind_id='k{}')  # one-hot `measurement` MPS.
#   bit_state2 = qmbt.MPS_product_state(arrays, site_ind_id='b{}')  # one-hot `measurement` MPS.

#   # assert (mps2 | bit_state) ^ ... == mps2 @ bit_state
#   # np.testing.assert_allclose((mps2 | bit_state) ^ ...,  mps1.amplitude(basis[:3] % 2))
#   np.testing.assert_allclose((mps2 | bit_state2) ^ ...,  (mps1 | bit_state1) ^ ...)

# test_mps_mpo_contraction()
