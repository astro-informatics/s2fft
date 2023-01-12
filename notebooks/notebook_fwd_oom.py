# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys

sys.path.append("../")

import healpy as hpy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pyssht as ssht
from jax import device_put, jit
from jax.config import config

import s2fft as s2f
import s2fft.healpix_ffts as hp
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.samples as samples
import s2fft.wigner as wigner

config.update("jax_enable_x64", True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data

# input params
L = 5  # 128 # in tests: 5
spin = 0  # 2 # in tests: [0, 1, 2] # should be 0 if reality=True
sampling = "mw"  #'dh' # in tests: ["mw", "mwss", "dh", "healpix"]
nside = 2  # 2,4,8, if healpix, else None 
if nside:
    L = 2 * nside  # 2 or 3 in tests
reality = True # reality is set to false if Healpix and method is not direct or sov
L_lower = 2  # in tests: L_lower_to_test = [0, 1, 2]


# generate spherical harmonics (ground truth)
# random---modify to use JAX random key approach?
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)


# compute signal in time domain with ssht (starting point)
# if healpix: we use the inverse from 'direct'? (or 'sov_fft_vectorized'?)
if sampling == "healpix":
    flm_gt0 = s2f.utils.generate_flm(
        rn_gen, L, L_lower, spin, reality=reality
    )  # shape L, 2L-1
    f = s2f.transform._inverse(
        flm_gt0, L, sampling=sampling, method="direct", nside=nside
    )
    flm_gt = hpy.sphtfunc.map2alm(np.real(f), lmax=L - 1, iter=0)  # ground-truth
    flm_gt = s2f.samples.flm_hp_to_2d(flm_gt, L)

else:
    flm_gt = s2f.utils.generate_flm(
        rn_gen, L, L_lower, spin, reality=reality
    )  # shape L, 2L-1
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
flm_sov_fft_vec = s2f.transform._forward(
    f,
    L,
    spin,
    sampling,
    method=method_str,
    nside=nside,
    reality=reality,
    L_lower=L_lower,
)

print(f'{sampling}, numpy vs GT: {np.allclose(flm_gt, flm_sov_fft_vec, atol=1e-14)}')  # False if healpix

%timeit s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside, reality=reality, L_lower=L_lower)
# 2.62 ms ± 4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %%%%%%%%%%%%%%%%%%%%%%
# All JAX methods
list_method_str = [ 
        "jax_vmap_double",
        "jax_vmap_scan",
        "jax_vmap_scan_0",
        "jax_vmap_loop",
        "jax_vmap_loop_0",
]
for m_str in list_method_str: 
    flm_sov_fft_vec_jax = s2f.transform._forward(
        f,
        L,
        spin,
        sampling,
        method=m_str,
        nside=nside,
        reality=reality,
        L_lower=L_lower,
    )

    print(m_str)
    if sampling=='healpix': # compare to numpy
        print(f'{sampling}, JAX vs numpy: {np.allclose(flm_sov_fft_vec, flm_sov_fft_vec_jax, atol=1e-14)}') 
    else:
        print(f'{sampling}, JAX vs GT: {np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14)}') 

    %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
    %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
    print('----')


#######################################