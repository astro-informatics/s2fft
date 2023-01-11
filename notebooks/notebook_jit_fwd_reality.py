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
sampling = "healpix"  #'dh' # in tests: ["mw", "mwss", "dh", "healpix"]
nside = 4  # 2,4,8, if healpix, else None 
if nside:
    L = 2 * nside  # 2 or 3 in tests
reality = True
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

# %timeit s2f.transform._forward(f, L, spin, sampling, method=method_str, nside=nside)
# 3.19 ms ± 28.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Using SOV + FFT Vectorised JAXed w JIT + vmap
method_str = "sov_fft_vectorized_jax_vmap"
flm_sov_fft_vec_jax = s2f.transform._forward(
    f,
    L,
    spin,
    sampling,
    method=method_str,
    nside=nside,
    reality=reality,
    L_lower=L_lower,
)

# plt.figure()
# plt.matshow(abs(flm_gt)) 
# plt.title('GT')
# plt.colorbar()
# plt.matshow(abs(flm_sov_fft_vec_jax)) 
# plt.title('JAX')
# plt.colorbar()
# plt.matshow(abs(flm_gt-flm_sov_fft_vec_jax)) 
# plt.colorbar()
# plt.title('differences')


if sampling=='healpix': # compare to numpy
    print(f'{sampling}, JAX vs numpy: {np.allclose(flm_sov_fft_vec, flm_sov_fft_vec_jax, atol=1e-14)}') 
else:
    print(f'{sampling}, JAX vs GT: {np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14)}') 




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare dl full matrix Numpy vs JAX
# if reality and spin != 0:
#     reality = False
#     warn(
#         "Reality acceleration only supports spin 0 fields. "
#         + "Defering to complex transform."
#     )

if sampling.lower() == "healpix" and method_str not in ["direct", "sov"]:
    reality = False

if sampling.lower() == "mw":
    f = resampling.mw_to_mwss(f, L, spin)

if sampling.lower() in ["mw", "mwss"]:
    sampling = "mwss"
    f = resampling.upsample_by_two_mwss(f, L, spin)
    thetas = samples.thetas(2 * L, sampling)

else:
    thetas = samples.thetas(L, sampling, nside)

# Don't need to include spin in weights (even for spin signals)
# since accounted for already in periodic extension and upsampling.
weights = quadrature.quad_weights_transform(L, sampling, 0, nside)

dl_full = np.zeros(
    (len(thetas), 
    len(range(max(L_lower, abs(spin)), L)),
    2*L-1)
    )
dl_full_jax = jnp.zeros(
    (len(thetas), 
    len(range(max(L_lower, abs(spin)), L)),
    2*L-1)
    )
for t, theta in enumerate(thetas):
    for el_i, el in enumerate(range(max(L_lower, abs(spin)), L)):

            # Turok Numpy with reality
            dl_full[t,el_i,:] = wigner.turok.compute_slice(theta, el, L, -spin, reality) # returns array of size [2L-1,]

            # OJO Turok_JAX implementation returns some garbage elements....only the slice [L-1-el:L-1+el+1] matches
            # we fill only the correct values and rest w zeroes
            dl_full_jax = dl_full_jax.at[t, el_i, L-1-el:L-1+el+1].set(
                wigner.turok_jax.compute_slice(theta, el, L, -spin, reality)[L-1-el:L-1+el+1]
                )

np.allclose(dl_full,dl_full_jax,atol=1e-14)

# %%##########################
# Compare dl vmapped vs numpy

# match shape
dl_full_trans = jnp.moveaxis(dl_full,0,-1)

# Define dl_vmapped fn
dl_vmapped = jax.vmap(
    jax.vmap(
        wigner.turok_jax.compute_slice,  # (theta, el, L, -spin, reality)
        in_axes=(0, None, None, None, None),
        out_axes=-1,
    ),
    in_axes=(None, 0, None, None, None),
    out_axes=0,
)

# Compute dl_vmp
dl_vmp = dl_vmapped(
    thetas, 
    jnp.arange(max(L_lower, abs(spin)), L),
    L,
    -spin, 
    reality)

# Check differences before masking
plt.figure()
plt.matshow(abs(dl_vmp[:,:,:]-dl_full_trans[:,:,:]).sum(axis=-1)) # abs(dl_vmp[:,:,:]-dl_full_trans[:,:,-1])
plt.colorbar()
plt.title('differences before masking')

# Mask output from dl_vmp
dl_vmp_padded = jnp.pad(dl_vmp, ((max(L_lower, abs(spin)), 0), (0, 0), (0,0)))  # TODO: Do I need abs(spin)? check
upper_diag = jnp.triu(jnp.ones_like(dl_vmp_padded, dtype=bool).T, k=-(L - 1)).T
mask = upper_diag * jnp.fliplr(upper_diag)
plt.matshow(mask[max(L_lower, abs(spin)):,:,1])
dl_vmp *= mask[max(L_lower, abs(spin)):,:,:]

# Compare to ground-truth after masking
print(np.allclose(dl_full_trans,dl_vmp,atol=1e-14))

plt.figure()
plt.matshow(abs(dl_vmp[:,:,:]-dl_full_trans[:,:,:]).sum(axis=-1)) # abs(dl_vmp[:,:,:]-dl_full_trans[:,:,-1])
plt.colorbar()
plt.title('differences after masking')

