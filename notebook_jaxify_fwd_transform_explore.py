"""
Check jaxify_transform.ipynb for timeit

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

# print(f"flm GT = {flm_gt}") # 2D complex128 np array of shape: (128, 255) (= (L, 2L-1))
# print(f"f = {f}") # shape: (256, 255)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Inspect flm
print(flm_gt.shape)
print(type(flm_gt))
print(flm_gt.dtype)

print(flm_gt.real.max())
print(flm_gt.real.min())

print(flm_gt.imag.max())
print(flm_gt.imag.min())
# print(flm.diagonal())

plt.matshow(flm_gt.real)
plt.colorbar()
plt.title("flm real part")
plt.matshow(flm_gt.imag)
plt.colorbar()
plt.title("flm imag part")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Inspect f
print(f.shape)  # (2L, 2L-1)?

print(type(f))
print(f.dtype)

print(f.real.max())
print(f.real.min())

print(f.imag.max())
print(f.imag.min())
# print(flm.diagonal())

plt.matshow(f.real)  # plt.matshow(np.abs(f.real))
plt.colorbar()
plt.title("f real part")
plt.matshow(f.imag)
plt.colorbar()
plt.title("f imag part")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare result from ssht toolbox (CPython) to GT
flm_ssht = ssht.forward(
    f, L, Spin=spin, Method=sampling, Reality=False  # OJO default sampling is MW
)
print(flm_ssht.shape)
print(s2f.samples.flm_1d_to_2d(flm_ssht, L).shape)
print(flm_gt.shape)

# OJO allclose is True if: absolute(a - b) <= (atol + rtol * absolute(b))
print(np.allclose(flm_gt, s2f.samples.flm_1d_to_2d(flm_ssht, L), atol=1e-14))

# plot diffs in real part
plt.matshow(abs(flm_gt.real - s2f.samples.flm_1d_to_2d(flm_ssht, L).real))
plt.colorbar()
# plt.matshow(1e-14 + 1e-05*abs(s2f.samples.flm_1d_to_2d(flm_ssht,L).real)) # this should be larger than the abs diff
# plt.colorbar()

plt.matshow(abs(flm_gt.imag - s2f.samples.flm_1d_to_2d(flm_ssht, L).imag))
plt.colorbar()

# Timeit
# %timeit flm_ssht = ssht.forward(f, L, Spin=spin, Method=sampling, Reality=False)
# 18.1 ms ± 254 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ### Using direct with SOV + FFT Vectorised (*closest to jax)
flm_sov_fft_vec = s2f.transform.forward_sov_fft_vectorized(f, L, spin, sampling)

print(np.allclose(flm_gt, flm_sov_fft_vec, atol=1e-14))

# %timeit s2f.transform.forward_sov_fft_vectorized(f, L, spin, sampling)
# 19.1 s ± 172 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ### Using direct with SOV + FFT Vectorised + JAX TUROK ---FALSE OJO!
flm_sov_fft_vec_jax_turok = s2f.transform.forward_sov_fft_vectorized_jax_turok(
    f, L, spin, sampling
)

print(np.allclose(flm_gt, flm_sov_fft_vec_jax_turok, atol=1e-14))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Exploring how to JAXIFY
import numpy.fft as fft
import jax.numpy.fft as jfft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner


# transform to DeviceArrays and put all arrays in default device?
if sampling.lower() == "mw":
    f = resampling.mw_to_mwss(f, L, spin)

if sampling.lower() in ["mw", "mwss"]:
    sampling = "mwss"
    f = resampling.upsample_by_two_mwss(f, L, spin)
    thetas = samples.thetas(2 * L, sampling)
else:
    thetas = samples.thetas(L, sampling)


# transform f to DeviceArray and commit to device---all the rest DeviceArrays and committed too? check
device = jax.devices()[0]
f = jax.device_put(f, device)
thetas = jax.device_put(thetas)  # float64

# initialise flm and ftm matrices and commit to device
# flm = jax.device_put(jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128), device) # initialise array and put in device
ftm = jnp.fft.fftshift(
    jnp.fft.fft(f, axis=1, norm="backward"), axes=1
)  # FFT to input signal? Use JAX implementation!--should be in device already?

# Don't need to include spin in weights (even for spin signals)
# since accounted for already in periodic extension and upsampling.
weights = jax.device_put(
    quadrature.quad_weights_transform(L, sampling, spin=0), device
)  # put in device? dtype?
m_offset = 1 if sampling == "mwss" else 0


el_array = jnp.array(
    range(spin, L), dtype=np.int64
)  # needs to be int for wigner.turok_jax.compute_slice
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define flm_summands function
# this will be the fn to vmap to receive 3D arrays? --I dont think I need to vmap, its just elementwise multiplication
# def flm_summands(w, el_f, dl_vector, ftm_vector):  # scalar  # scalar  # 1D array)
#     '''
#     Returns flm summands along the last axis
#     '''
#     return w * el_f * dl_vector * ftm_vector
flm_summands = lambda w, el_f, dl_vector, ftm_vector: w * el_f * dl_vector * ftm_vector

# # check
# t=0
# theta = thetas[t]
# e=0
# el = list(range(spin,L))[e]

# elfactor = np.sqrt((2 * el + 1) / (4 * np.pi))
# dl = np.zeros((len(thetas),len(range(spin, L)),2*L-1), dtype=np.float64)
# for t, theta in enumerate(thetas):
#     for e, el in enumerate(range(spin, L)):
#         dl[t,e,:] = wigner.turok_jax.compute_slice(theta, el, L, -spin) #--vmap for all thetas? and all el

# np.allclose(weights[0]* elfactor * dl[t,e] * ftm[t, m_offset : 2 * L - 1 + m_offset],
#             flm_summands(weights[0], elfactor, dl[t,e], ftm[t, m_offset : 2 * L - 1 + m_offset]))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute weights as 3D array
target_size_3D = (
    len(range(spin, L)),  # samples.flm_shape(L)[0]-spin?
    samples.flm_shape(L)[1],
    len(thetas),
)
weights_3D = jnp.broadcast_to(weights, target_size_3D)

print(weights_3D.shape)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute el_factor as 3D array
el_array_fl = jnp.array(
    el_array, dtype=np.float64
)  # to prevent el_factor_3D.weak_type=True
el_factor_3D = jnp.broadcast_to(
    (((2 * el_array_fl + 1) / (4 * jnp.pi)) ** 0.5)[:, None, None], target_size_3D
)
print(el_factor_3D.shape)

# maybe no need to broadcast view?
# np.allclose(weights_3D * (((2 * el_array + 1) / (4 * jnp.pi))**0.5)[:,None,None],
#             weights_3D * el_factor_3D)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute dl 3D array
# dl vmapped fn returns an array with :
# - len(range(spin,L)) rows (axis 0, 'el')
# - 2L-1 columns (axis 1)
# - 2L channels (axis 2, 'theta') ---OJO at axes order! different from fn inputs
dl_vmapped = jax.vmap(
    jax.vmap(
        wigner.turok_jax.compute_slice,  # wigner.turok_jax.compute_slice(theta, el, L, -spin)
        in_axes=(0, None, None, None),
        out_axes=-1,
    ),
    in_axes=(None, 0, None, None),
    out_axes=0,
)
dl_3D = dl_vmapped(thetas, el_array, L, -spin)
print(dl_3D.shape)
# dl_vmapped(thetas,el_array,L,-spin)[0,:,0] == dl[0,0] == dl at theta=0, el=0 (returns a vector of len 2L-1)
# OJO! Size changes with spin

# Pad with zeros?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute ftm 3D array
# ftm[t, m_offset : 2 * L - 1 + m_offset]
ftm_3D = jnp.broadcast_to(ftm[:, m_offset : 2 * L - 1 + m_offset].T, target_size_3D)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check summands
# could be jit-compiled?
result = (
    flm_summands(
        weights_3D, el_factor_3D, dl_3D, ftm_3D  # scalar  # scalar  # 1D array)
    )
    .sum(axis=-1)
    .at[:]
    .multiply((-1) ** spin)
)

# result = result.at[:].multiply((-1) ** spin)
print(result.shape)

# compare to alternative computation of flm: flm_sov_fft_vec_jax_turok
print(np.allclose(result, 
                 flm_sov_fft_vec_jax_turok[spin:L, :], 
                 atol=1e-14))
# ...
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Alternative --directly broadcasting (any better?)
result2 = (
    flm_summands(
        weights[None, None, :],
        (((2 * el_array_fl + 1) / (4 * jnp.pi)) ** 0.5)[:, None, None],
        dl_vmapped(thetas, el_array, L, -spin),
        ftm[:, m_offset : 2 * L - 1 + m_offset, None].T,
    )
    .sum(axis=-1)
    .at[:]
    .multiply((-1) ** spin)
)

result2 = jnp.pad(result2, ((spin,0),(0,0)))

print(result2.shape)
print(np.allclose(result2, 
                  flm_sov_fft_vec_jax_turok, #[spin:L, :],
                  atol=1e-14))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Alternative --using stack (worse for memory?)
result3 = (
    jnp.stack((weights_3D, el_factor_3D, dl_3D, ftm_3D), axis=-1)
    .prod(axis=-1)
    .sum(axis=-1)
    .at[:]
    .multiply((-1) ** spin)
)

#   .sum(axis=-1).at[:].multiply((-1)**spin)

print(result3.shape)
print(np.allclose(result, 
                  result3, 
                  atol=1e-14))

# %%

# %%
