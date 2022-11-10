
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

import numpy.fft as fft
import jax.numpy.fft as jfft
import s2fft.samples as samples
import s2fft.quadrature as quadrature
import s2fft.resampling as resampling
import s2fft.wigner as wigner

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Sample data
# alternative: earth?

# input params
L = 5 #128 # in tests: 5
spin = 2 #2 # in tests: [0, 1, 2]
sampling = 'dh' # in tests: ["mw", "mwss", "dh"]

# generate spherical harmonics (ground truth)
# random---modify to use JAX random key approach?
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)
flm_gt = s2f.utils.generate_flm(rn_gen, L, spin, reality=False) # flm_2d = flm_1d_to_2d(flm, L) # groundtruth sph harmonics?

# compute signal in time domain (starting point)
f = ssht.inverse(s2f.samples.flm_2d_to_1d(flm_gt, L), # 2D indexed coeffs to 1D indexed
                 L,
                 Method=sampling.upper(),
                 Spin=spin,
                 Reality=False) # use ssht to compute signal in time domain-- starting point

# print(f"flm GT = {flm_gt}") # 2D complex128 np array of shape: (128, 255) (= (L, 2L-1)) 
# print(f"f = {f}") # shape: (256, 255)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

device = jax.devices()[0]

# transform f to DeviceArray and commit to device---all the rest DeviceArrays and committed too? check
f = jax.device_put(f, device)

# transform to DeviceArrays and put all arrays in default device?
if sampling.lower() == "mw":
    f = resampling.mw_to_mwss(f, L, spin)

if sampling.lower() in ["mw", "mwss"]:
    sampling = "mwss"
    f = resampling.upsample_by_two_mwss(f, L, spin)
    thetas = samples.thetas(2 * L, sampling)
else:
    thetas = samples.thetas(L, sampling)
thetas = jax.device_put(thetas) #float64

# initialise flm and ftm matrices and commit to device
flm = jax.device_put(jnp.zeros(samples.flm_shape(L), dtype=jnp.complex128), device) # initialise array and put in device
ftm = jnp.fft.fftshift(jnp.fft.fft(f, axis=1, norm="backward"), axes=1) #FFT to input signal? Use JAX implementation!--should be in device already?

# Don't need to include spin in weights (even for spin signals)
# since accounted for already in periodic extension and upsampling.
weights = jax.device_put(quadrature.quad_weights_transform(L, sampling, spin=0), device) #put in device? dtype?
m_offset = 1 if sampling == "mwss" else 0


el_array = jnp.array(range(spin, L), dtype=np.int64)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Computing dl matrix
###################################################
# %%
# reference naive dl matrix
dl = np.zeros((len(thetas),len(range(spin, L)),2*L-1), dtype=np.float64)
for t, theta in enumerate(thetas):

    for e, el in enumerate(range(spin, L)):

        dl[t,e,:] = wigner.turok_jax.compute_slice(theta, el, L, -spin) #--vmap for all thetas? and all el

print(dl.shape)
print(type(dl))

# do I need to flip?
# see https://github.com/astro-informatics/s2fft/blob/main/tests/test_wigner.py
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# JAXed dl matrix: using vmap
# thetas = jax.device_put(thetas) #float64 # DeviceArray
el_array = jnp.array(range(spin, L), dtype=np.int64)

# wigner.turok_jax.compute_slice(theta, el, L, -spin)
dl_vmapped_theta = jax.vmap(wigner.turok_jax.compute_slice, 
                            in_axes=(0,None,None,None),
                            out_axes=0)

dl_vmapped = jax.vmap(dl_vmapped_theta, 
                      in_axes=(None,0,None,None),
                      out_axes=1)

print(dl_vmapped(thetas,el_array,L,-spin).shape)

# Check vmapping is correct
jnp.max(jnp.abs(jnp.array(dl) - dl_vmapped(thetas,el_array,L,-spin)))

print(np.allclose(dl,
                  dl_vmapped(thetas,el_array,L,-spin),
                  atol=1e-14))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check alternative axes orientation

print(dl_vmapped(thetas,el_array,L,-spin).shape)
print(jnp.moveaxis(dl_vmapped(thetas,el_array,L,-spin),[0,1,2],[2,0,1]).shape)


# wigner.turok_jax.compute_slice(theta, el, L, -spin)
dl_vmapped2 = jax.vmap(jax.vmap(wigner.turok_jax.compute_slice, 
                                in_axes=(0,None,None,None),
                                out_axes=-1),
                        in_axes=(None,0,None,None),
                        out_axes=0)

print(dl_vmapped2(thetas,el_array,L,-spin).shape)

print(np.allclose(jnp.moveaxis(dl_vmapped(thetas,el_array,L,-spin),[0,1,2],[2,0,1]),
                  dl_vmapped2(thetas,el_array,L,-spin),
                  atol=1e-14))
# %%
