from s2fft.utils.healpix_ffts import (
    healpix_fft_jax,
    healpix_fft_numpy,
    healpix_ifft_jax,
    healpix_ifft_numpy,
)
import healpy as hp

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import s2fft 
import time
from itertools import product

nside = 4
L = 2 * nside
method = "jax"
sampling = "healpix"

total_pxiels = nside**2 * 12

healpix_array = jax.random.normal(jax.random.PRNGKey(0), (total_pxiels,))
print(f"Shape of healpix_array = {healpix_array.shape}")

start = time.perf_counter()
alm = healpix_fft_jax(healpix_array, L, nside , False).block_until_ready()
end = time.perf_counter()
print(f"JIT Time taken for healpix_fft_jax = {(end - start)*1000} ms")
start = time.perf_counter()
alm = healpix_fft_jax(healpix_array, L, nside , False)
end = time.perf_counter()
print(f"Time taken for healpix_fft_jax = {(end - start)*1000} ms")

#start = time.perf_counter()
#healpix_array_recov = healpix_ifft_jax(alm, L, nside , False).block_until_ready()
#end = time.perf_counter()
#print(f"JIT Time taken for healpix_ifft_jax = {(end - start)*1000} ms")
#start = time.perf_counter()
#healpix_array_recov = healpix_ifft_jax(alm, L, nside , False).block_until_ready()
#end = time.perf_counter()
#print(f"Time taken for healpix_ifft_jax = {(end - start)*1000} ms")
#
#print(f"Max error = {jnp.max(jnp.abs(healpix_array_recov - healpix_array))}")
print(f"Shape of alm = {alm}")
for i , a in enumerate(alm):
    print(f"Alm[{i}] = {a}")