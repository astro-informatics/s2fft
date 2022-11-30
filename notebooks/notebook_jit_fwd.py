# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
sys.path.append('../')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

import pyssht as ssht
import s2fft as s2f
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner
import s2fft.healpix_ffts as hp
import healpy as hpy

import jax
from jax import jit, device_put
import jax.numpy as jnp
from jax.config import config

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data

# input params
L = 5  # 128 # in tests: 5
spin = 2  # 2 # in tests: [0, 1, 2]
sampling = "mw"  #'dh' # in tests: ["mw", "mwss", "dh", "healpix"]
nside = None #[2,4,8] in tests, None if sampling not healpix

# generate spherical harmonics (ground truth)
# random---modify to use JAX random key approach?
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)


# compute signal in time domain with ssht (starting point)
# if healpix: we use the inverse from 'direct'? (or 'sov_fft_vectorized'?)
if sampling == 'healpix':
    flm_gt0 = s2f.utils.generate_flm(rn_gen, L, spin, reality=False) # shape L, 2L-1
    f = s2f.transform._inverse(flm_gt0, L, sampling=sampling, method='direct', nside=nside)
    flm_gt = hpy.sphtfunc.map2alm(np.real(f), lmax=L - 1, iter=0) #ground-truth
    flm_gt = s2f.samples.flm_hp_to_2d(flm_gt, L)
    
else:
    flm_gt = s2f.utils.generate_flm(rn_gen, L, spin, reality=False) # shape L, 2L-1
    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm_gt, L),  # 2D indexed coeffs to 1D indexed
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Using SOV + FFT Vectorised (Numpy)
method_str = "sov_fft_vectorized"
flm_sov_fft_vec = s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside)

print(np.allclose(flm_gt, flm_sov_fft_vec, atol=1e-14)) #returns False for healpix

%timeit s2f.transform._forward(f, L, spin, sampling, method=method_str)
# 3.19 ms ± 28.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Using SOV + FFT Vectorised JAXed w JIT
method_str = "sov_fft_vectorized_jax"
flm_sov_fft_vec_jax = s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside)

print(np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14)) #returns False for healpix

%timeit s2f.transform._forward(f, L, spin, sampling, method=method_str)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Using SOV + FFT Vectorised JAXed (no JIT)  ---only transform

# if sampling.lower() == "mw":
#     f = resampling.mw_to_mwss(f, L, spin)

# if sampling.lower() in ["mw", "mwss"]:
#     sampling = "mwss"
#     f = resampling.upsample_by_two_mwss(f, L, spin)
#     thetas = samples.thetas(2 * L, sampling)

# else:
#     thetas = samples.thetas(L, sampling, nside)

# weights = quadrature.quad_weights_transform(L, sampling, 0, nside)


# flm_sov_fft_vec_jax = s2f.transform._compute_forward_sov_fft_vectorized_jax(f, L, spin, sampling, thetas, weights, nside) # shape L, 2L-1

# print(np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14))

# %timeit s2f.transform._compute_forward_sov_fft_vectorized_jax(f, L, spin, sampling, thetas, weights, nside).block_until_ready()


# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Using SOV + FFT Vectorised JAXed w JIT ---only transform
# if sampling.lower() == "mw":
#     f = resampling.mw_to_mwss(f, L, spin)

# if sampling.lower() in ["mw", "mwss"]:
#     sampling = "mwss"
#     f = resampling.upsample_by_two_mwss(f, L, spin)
#     thetas = samples.thetas(2 * L, sampling)

# else:
#     thetas = samples.thetas(L, sampling, nside)

# weights = quadrature.quad_weights_transform(L, sampling, 0, nside)

# ###
# forward_transform_jax_jitted = jit(s2f.transform._compute_forward_sov_fft_vectorized_jax, 
#                                    static_argnums=(1, 2, 3, 6))

# flm_sov_fft_vec_jax_jit = forward_transform_jax_jitted(f, L, spin, sampling, thetas, weights, nside) # shape L, 2L-1

# print(np.allclose(flm_gt, flm_sov_fft_vec_jax_jit, atol=1e-14))

# %timeit forward_transform_jax_jitted(f, L, spin, sampling, thetas, weights, nside).block_until_ready()
# # 29.6 µs ± 128 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

# %%
