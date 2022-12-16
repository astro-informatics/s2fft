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
import jax.numpy.fft as jfft


import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data

# input params
L = 5  # 128 # in tests: 5
spin = 2  # 2 # in tests: [0, 1, 2]
sampling = "healpix"  #'dh' # in tests: ["mw", "mwss", "dh", "healpix"]
nside = 2 #2,4,8 None if sampling not healpix
if nside:
    L = 2*nside # 2 or 3 in tests

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Spectral periodic extension: compare jax and non-jax

# from healpix_fft
ftm = jnp.zeros(samples.ftm_shape(L, "healpix", nside), # shape = ntheta, 2L
                dtype=jnp.complex128)

index = 0
for t in range(ftm.shape[0]):
    nphi = samples.nphi_ring(t, nside)
    fm_chunk = jfft.fftshift(jfft.fft(f[index : index + nphi], norm="backward"))
    spectral_out_1 = hp.spectral_periodic_extension(fm_chunk, nphi, L)
    spectral_out_2 = hp.spectral_periodic_extension_jax(fm_chunk, L, jnp) #(fm_chunk, L, jnp)
    print(np.allclose(
        spectral_out_1, 
        spectral_out_2, 
        atol=1e-14)) 
    # ftm = ftm.at[t].set(spectral_out_1)
    index += nphi


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check healpix FFT JAX
# first approach
ftm_1 = hp.healpix_fft_jax_1(f, L, nside)

# lax scan approach
ftm_2 = hp.healpix_fft_jax_2(f, L, nside)

# jax.numpy/numpy approach (Matt G)
ftm_3 = hp.healpix_fft_jax_3(f, L, nside, jnp)

# vmap approach

print(np.allclose(ftm_1, ftm_2, atol=1e-14)) # False because of padding with 0s! t=0 and t=end are different
print(np.allclose(ftm_1, ftm_3, atol=1e-14))

# %%
plt.figure()
plt.matshow(ftm_1.imag-ftm_2.imag)
plt.title('imag')
plt.colorbar()

plt.figure()
plt.matshow(ftm_1.real-ftm_2.real)
plt.title('R')
plt.colorbar()
# %%
