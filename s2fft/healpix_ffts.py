import numpy as np
import numpy.fft as fft
import s2fft.samples as samples

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

def spectral_folding(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. aliasing high frequencies.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Lower resolution set of aliased Fourier coefficients.
    """
    assert nphi <= 2 * L

    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    ftm_slice = fm[slice_start:slice_stop]

    idx = 1
    while slice_start - idx >= 0:
        ftm_slice[-idx % nphi] += fm[slice_start - idx]
        idx += 1
    idx = 0
    while slice_stop + idx < len(fm):
        ftm_slice[idx % nphi] += fm[slice_stop + idx]
        idx += 1

    return ftm_slice


def spectral_periodic_extension(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.
    """
    assert nphi <= 2 * L

    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    fm_full = np.zeros(2 * L, dtype=np.complex128)
    fm_full[slice_start:slice_stop] = fm

    idx = 1
    while slice_start - idx >= 0:
        fm_full[slice_start - idx] = fm[-idx % nphi]
        idx += 1
    idx = 0
    while slice_stop + idx < len(fm_full):
        fm_full[slice_stop + idx] = fm[idx % nphi]
        idx += 1

    return fm_full

def spectral_periodic_extension_jax(fm: np.ndarray, nphi: int, L: int) -> np.ndarray:
    """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on `spectral_periodic_extension`,
    modified to be JIT-compilable.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.
    """
    assert nphi <= 2 * L

    slice_start = L - nphi // 2
    slice_stop = slice_start + nphi
    fm_full = jnp.zeros(2 * L, dtype=np.complex128)
    fm_full = fm_full.at[slice_start:slice_stop].set(fm)

    idx = 1
    while slice_start - idx >= 0:
        fm_full = fm_full.at[slice_start - idx].set(fm[-idx % nphi])
        idx += 1
    idx = 0
    while slice_stop + idx < len(fm_full):
        fm_full = fm_full.at[slice_stop + idx].set(fm[idx % nphi])
        idx += 1

    return fm_full

def healpix_fft(f: np.ndarray, L: int, nside: int) -> np.ndarray:
    """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    assert L >= 2 * nside

    index = 0
    ftm = np.zeros(samples.ftm_shape(L, "healpix", nside), dtype=np.complex128)
    ntheta = ftm.shape[0]
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = fft.fftshift(fft.fft(f[index : index + nphi], norm="backward"))
        ftm[t] = (
            fm_chunk
            if nphi == 2 * L
            else spectral_periodic_extension(fm_chunk, nphi, L)
        )
        index += nphi
    return ftm

def healpix_fft_jax(f: np.ndarray, L: int, nside: int) -> np.ndarray:
    """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity. Based on `healpix_fft`,
    modified to be JIT-compilable.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
    # Check Matt G's suggestion https://github.com/astro-informatics/s2fft/pull/128#discussion_r1038152725

    # assert L >= 2 * nside

    # # theta index array
    # t_array = jnp.expand_dims(jnp.arange(samples.ftm_shape(L, "healpix", nside)[0]),axis=-1) #, dtype=int)
    
    # # index 2D array
    # index_array = jnp.pad(jnp.cumsum(t_array, axis=0)[:-1,:],((1, 0), (0, 0)))  

    # # nphi 2D array
    # nphi_ring_vmapped = jax.vmap(
    #     samples.nphi_ring_jax, # ----------needs to be jaxed too!
    #     in_axes=(0,None),
    #     out_axes=0
    # )
    # nphi_array = nphi_ring_vmapped(t_array, nside)
    
    # # fm_chunk 2D array (can I omit it?)
    # get_fm_chunk = lambda f,ix,nph: jfft.fftshift(
    #     jfft.fft(
    #         f[ix : ix + nph], 
    #         norm="backward"
    #         )
    #         ) 
    # get_fm_chunk_vmapped = jax.vmap(
    #     get_fm_chunk,
    #     in_axes=(None,0,0),
    #     out_axes=0
    # )
    # fm_chunk_array = get_fm_chunk_vmapped(f, index_array, nphi_array)


    # # # spectral_periodic_extension_jax_array vmap?
    # spectral_periodic_extension_jax_vmapped = jax.vmap(
    #     spectral_periodic_extension_jax,
    #     in_axes=(0,0,None),
    #     out_axes=0
    # )
    # spectral_periodic_extension_jax_array = spectral_periodic_extension_jax_vmapped(
    #     fm_chunk_array,
    #     nphi_array,
    #     L
    # )


    # # # ftm
    # ftm = jnp.where(
    #     nphi_array==2*L, #nphi_ring_vmapped(t_array, nside)==2*L, 
    #     fm_chunk_array,
    #     spectral_periodic_extension_jax_array)
    
    # -------
    # loop thru thetas
    assert L >= 2 * nside
    ftm = jnp.zeros(samples.ftm_shape(L, "healpix", nside), # shape = ntheta, 2L
                    dtype=jnp.complex128)

    index = 0
    for t in range(ftm.shape[0]):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = jfft.fftshift(jfft.fft(f[index : index + nphi], norm="backward"))
        ftm = ftm.at[t].set(
            fm_chunk
            if nphi == 2 * L
            else spectral_periodic_extension_jax(fm_chunk, nphi, L)
        )
        index += nphi
    return ftm

def healpix_ifft(ftm: np.ndarray, L: int, nside: int) -> np.ndarray:
    """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
    assert L >= 2 * nside

    f = np.zeros(samples.f_shape(sampling="healpix", nside=nside), dtype=np.complex128)
    ntheta = ftm.shape[0]
    index = 0
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = ftm[t] if nphi == 2 * L else spectral_folding(ftm[t], nphi, L)
        f[index : index + nphi] = fft.ifft(fft.ifftshift(fm_chunk), norm="forward")
        index += nphi
    return f
