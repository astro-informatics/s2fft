"""
Check different JAX impl

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Imports
import numpy as np

import pyssht as ssht
import s2fft as s2f

import jax
from jax import jit, device_put
import jax.numpy as jnp
from jax.config import config

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data
# alternative: earth?

# input params
L = 5  # 128 # in tests: 5
spin = 2  # 2 # in tests: [0, 1, 2]
sampling = "mwss"  #'dh' # in tests: ["mw", "mwss", "dh"]

# generate spherical harmonics (ground truth)
# random---modify to use JAX random key approach?
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)
flm_gt = s2f.utils.generate_flm(
    rn_gen, L, spin, reality=False
)  # flm_2d = flm_1d_to_2d(flm, L) # groundtruth sph harmonics?

# compute signal in time domain (starting point)
f = ssht.inverse(
    s2f.samples.flm_2d_to_1d(flm_gt, L),  # 2D indexed coeffs to 1D indexed
    L,
    Method=sampling.upper(),
    Spin=spin,
    Reality=False,
)  # use ssht to compute signal in time domain-- starting point


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ### Using direct with SOV + FFT Vectorised (*closest to jax) ---FALSE OJO!
flm_sov_fft_vec = s2f.transform.forward_sov_fft_vectorized(f, L, spin, sampling)

print(np.allclose(flm_gt, flm_sov_fft_vec, atol=1e-14))

# %timeit s2f.transform.forward_sov_fft_vectorized(f, L, spin, sampling)
# 19.1 s ± 172 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ### Using direct with SOV + FFT Vectorised + JAX TUROK ---Use this as GT for now
# same as forward_sov_fft_vectorized but using wigner.turok_jax.compute_slice
flm_sov_fft_vec_jax_turok = s2f.transform.forward_sov_fft_vectorized_jax_turok(
    f, L, spin, sampling
)
# %timeit s2f.transform.forward_sov_fft_vectorized_jax_turok(f, L, spin, sampling)
print(np.allclose(flm_gt, flm_sov_fft_vec_jax_turok, atol=1e-14)) # ---Compared to GT gives FALSE OJO!

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# JAX approach 1
flm_jax_1 = s2f.transform.forward_sov_fft_vectorized_jax_1(
    f, L, spin, sampling
)
# compare to alternative computation of flm: flm_sov_fft_vec_jax_turok
print(np.allclose(flm_jax_1, 
                  flm_sov_fft_vec_jax_turok, #w/o padding: flm_sov_fft_vec_jax_turok[spin:L, :], 
                  atol=1e-14))



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# JAX approach 2
flm_jax_2 = s2f.transform.forward_sov_fft_vectorized_jax_2(
    f, L, spin, sampling
)
# compare to alternative computation of flm: flm_sov_fft_vec_jax_turok
print(np.allclose(flm_jax_2, 
                  flm_sov_fft_vec_jax_turok, #w/o padding: flm_sov_fft_vec_jax_turok[spin:L, :], 
                  atol=1e-14))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# JAX approach 3 (stacked)
flm_jax_3 = s2f.transform.forward_sov_fft_vectorized_jax_3(
    f, L, spin, sampling
)
# compare to alternative computation of flm: flm_sov_fft_vec_jax_turok
print(np.allclose(flm_jax_3, 
                  flm_sov_fft_vec_jax_turok, #w/o padding: flm_sov_fft_vec_jax_turok[spin:L, :], 
                  atol=1e-14))








# %%
