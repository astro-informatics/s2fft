# %%
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data

# input params
L = 4  # 128 # in tests: 5; if healpix: L = ratio * nside
spin = 2  # 2 # in tests: [0, 1, 2]
sampling = "healpix"  #'dh' # in tests: ["mw", "mwss", "dh", "healpix"]
nside = 2 #[2,4,8] in tests, None if sampling not healpix
ratio = 2

# generate spherical harmonics (ground truth)
# random---modify to use JAX random key approach?
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)


# compute signal in time domain with ssht (starting point)
# if healpix: we use the inverse from 'direct'? (or 'sov_fft_vectorized'?)
if sampling == 'healpix':
    L = ratio * nside
    flm_gen = s2f.utils.generate_flm(rn_gen, L, spin=0, reality=False) # shape L, 2L-1
    # flm_hp = s2f.samples.flm_2d_to_hp(flm_gen, L)
    # f_check = hpy.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)
    f = s2f.transform._inverse(flm_gen, L, sampling=sampling, method='direct', nside=nside)
    flm_gt = hpy.sphtfunc.map2alm(np.real(f), lmax=L - 1, iter=0)
else:
    flm_gt = s2f.utils.generate_flm(rn_gen, L, spin, reality=False) # shape L, 2L-1
    f = ssht.inverse(
        s2f.samples.flm_2d_to_1d(flm_gt, L),  # 2D indexed coeffs to 1D indexed
        L,
        Method=sampling.upper(),
        Spin=spin,
        Reality=False,
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Using SOV + FFT Vectorised ---- returns False for healpix :/
# method_str = "sov_fft_vectorized"
# flm_sov_fft_vec0 = s2f.transform._forward(f, L, sampling=sampling, method=method_str, nside=nside)

# if sampling=='healpix':
#     flm_sov_fft_vec = s2f.samples.flm_2d_to_hp(flm_sov_fft_vec0, L)

# print(np.allclose(flm_gt, flm_sov_fft_vec, atol=1e-14)) #returns False for healpix :/

# # %timeit s2f.transform._forward(f, L, spin, sampling, method=method_str)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Using SOV + FFT Vectorised JAXed (compare to numpy implementation)
method_str = "sov_fft_vectorized_jax"
flm_sov_fft_vec_jax = s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside) # shape L, 2L-1

method_str = "sov_fft_vectorized"
flm_sov_fft_vec_np = s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside)

print(np.allclose(flm_sov_fft_vec_np, flm_sov_fft_vec_jax, atol=1e-14))

# print(np.allclose(flm_gen, flm_sov_fft_vec_jax, atol=1e-14))




# print(np.allclose(flm_sov_fft_vec, flm_sov_fft_vec_jax, atol=1e-14)) 
# ---returns True if replacing `wigner.turok.compute_slice(theta, el, L, -spin)` with 
# `np.array(wigner.turok_jax.compute_slice(theta, el, L, -spin))` in _compute_forward_sov_fft_vectorized

# %timeit s2f.transform._forward(f, L, spin, sampling, method=method_str)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Phase shift for JAX --boiler plate

if sampling.lower() == "mw":
    f = resampling.mw_to_mwss(f, L, spin)

if sampling.lower() in ["mw", "mwss"]:
    sampling = "mwss"
    f = resampling.upsample_by_two_mwss(f, L, spin)
    thetas = samples.thetas(2 * L, sampling)
else:
    thetas = samples.thetas(L, sampling,nside=nside)

weights = quadrature.quad_weights_transform(L, sampling, spin, nside=nside)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Phase shift array per theta --manually
phase_shift_per_theta = np.zeros((len(thetas), 2*L-1),dtype=complex)
if sampling.lower() == "healpix":
    for t, theta in enumerate(thetas):
        phase_shift_per_theta[t,:] = samples.ring_phase_shift_hp(L, t, nside, forward=True)
                
phase_shift_per_theta.shape

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Phase shift array per theta --manually using aux fn
# define aux fn (vmap has issues with keyword args: # https://github.com/google/jax/issues/7465)
# Q for review: is there a more elegant way?
def ring_phase_shift_hp_aux(L, t, nside, forward_bool):
    return samples.ring_phase_shift_hp_vmappable(L, t, nside, forward=forward_bool)
# fn_aux = lambda L, t, nside, forward: samples.ring_phase_shift_hp(L, t, nside, forward=forward)

phase_shift_per_theta2 = np.zeros((len(thetas), 2*L-1),dtype=complex)
if sampling.lower() == "healpix":
    for t, theta in enumerate(thetas):
        phase_shift_per_theta2[t,:] = ring_phase_shift_hp_aux(L, t, nside, True)
                
phase_shift_per_theta2.shape
np.allclose(phase_shift_per_theta,phase_shift_per_theta2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute phase shift vmapped
# vmap(): JAX introduces abstract tracers for all positional arguments.
# See: https://jax.readthedocs.io/en/latest/faq.html#different-kinds-of-jax-values
# and https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError 
if sampling.lower() == "healpix":
    phase_shift_vmapped = jax.vmap(ring_phase_shift_hp_aux,
                                   in_axes=(None,0,None,None), 
                                   out_axes=-1) # OJO! theta along rows if out_axes=0
else:
    phase_shift = 1.0    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare to manual
phase_shift_2D = phase_shift_vmapped(L, jnp.array(range(len(thetas))), nside, True)
print(phase_shift_2D.shape) 

np.allclose(phase_shift_2D.T,
            phase_shift_per_theta,
            atol=1e-14)



