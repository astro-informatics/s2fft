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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare All JAX methods to ground truth

# generate spherical harmonics (ground truth)
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED) # modify to use JAX random key approach?


# list jax implementations
list_jax_str = [ 
        "jax_vmap_double",#  "jax_vmap_double_mmg",
        "jax_vmap_scan",
        "jax_vmap_scan_0",
        "jax_vmap_loop",
        "jax_vmap_loop_0",
        "jax_map_double",
        "jax_map_scan"
]

# list of sampling approaches
list_sampling = ["mw", "mwss", "dh", "healpix"]


### All JAX methods
for m_str in list_jax_str: 

    for sampling in list_sampling:

        ### input params
        L = 5  
        spin = 1 
        reality = True 
        L_lower = 2  
        # only applicable to healpix: 
        nside = 2 # (set to None if sampling not healpix)
        if nside:
            L = 2 * nside  

        ### Groundtruth and starting point f
        if sampling!='healpix':
            ## Set nside to None if not healpix
            nside=None

            # compute signal in time domain with ssht (starting point)
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
        else:
            flm_gt0 = s2f.utils.generate_flm(
                rn_gen, L, L_lower, spin, reality=reality
            )  # shape L, 2L-1
            f = s2f.transform._inverse(
                flm_gt0, L, sampling=sampling, method="direct", nside=nside
            )

        ### Compute numpy solution if sampling is healpix (used as GT)
        if sampling=='healpix': 
            flm_sov_fft_vec = s2f.transform._forward(
                f,
                L,
                spin,
                sampling,
                method="sov_fft_vectorized",
                nside=nside,
                reality=reality,
                L_lower=L_lower,)
   
        ### JAX implementation
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

        if sampling=='healpix': # compare to numpy result rather than GT
            print(f'{m_str} vs numpy ({sampling}): {np.allclose(flm_sov_fft_vec, flm_sov_fft_vec_jax, atol=1e-14)}') 
        else:
            print(f'{m_str} vs GT ({sampling}): {np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14)}') 

        # %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
        # %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
    print('----')

