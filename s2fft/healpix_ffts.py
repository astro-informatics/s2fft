import numpy as np
import numpy.fft as fft
import s2fft.samples as samples

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from functools import partial


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


def spectral_periodic_extension_jax(fm, L, numpy_module=np): 
    """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on `spectral_periodic_extension`,
    modified to be JIT-compilable.

    Args:
        fm (np.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        L (int): Harmonic band-limit.
        
        numpy_module: module to use, either numpy (np, default) or JAX's Numpy-like API (jnp)

    Returns:
        np.ndarray: Higher resolution set of periodic Fourier coefficients.
    """
    nphi = fm.shape[0] 
    return numpy_module.concatenate(
        ( 
            fm[-numpy_module.arange(L - nphi // 2, 0, -1) % nphi], 
            fm, 
            fm[numpy_module.arange(L - (nphi + 1) // 2) % nphi] 
        ) 
    )

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


def healpix_fft_jax_1(f: np.ndarray, L: int, nside: int) -> np.ndarray:  
    '''
    Healpix FFT naive JAX implementation

    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    '''

    assert L >= 2 * nside
    ftm = jnp.zeros(samples.ftm_shape(L, "healpix", nside), # (ntheta, 2L)
                    dtype=jnp.complex128)

    index = 0
    for t in range(ftm.shape[0]):
        nphi = samples.nphi_ring(t, nside) # int
        fm_chunk = jfft.fftshift(jfft.fft(f[index : index + nphi], norm="backward")) #nphi varies per t
        ftm = ftm.at[t].set(spectral_periodic_extension_jax(fm_chunk, L, jnp)) # extends fm_chunks to their max 2L length
        index += nphi
    return ftm

#########
def healpix_fft_jax_2(f: np.ndarray, L: int, nside: int) -> np.ndarray:
    '''
    Healpix FFT JAX implementation using lax.scan---the Fourier coeffs for the first and last latitude are off as it is!

    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    '''

    assert L >= 2 * nside

    ts = np.arange(samples.ftm_shape(L, "healpix", nside)[0]) # concrete- list(range(samples.ftm_shape(L, "healpix", nside)[0])) #
    nphis = np.array([samples.nphi_ring(t, nside) for t in ts]) # concrete-[samples.nphi_ring(t, nside) for t in ts] #
    
    # need concrete for slicing
    ### list of chunks
    # f_chunks = [
    #     f[idx:idx + nphi] 
    #     for (idx,nphi) in zip(indices,nphis)
    #     ] 

    ### array of padded f_chunks
    indices = np.concatenate([[0],np.cumsum(nphis, axis=0)[:-1]]) # concrete-
    nphi_max = np.max(nphis)
    f_chunks_padded = jnp.stack(
        [
        jnp.pad(
            f[idx:idx + nphi], #nphi varies with t; pad with nans to make sizes consistent?
            ((0,nphi_max - nphi)),
            constant_values=0.0,
            ) 
            for (idx,nphi) in zip(indices,nphis)] 
    )

    def accumulate(ftm, ts_fchunks_tuple): 
        t, f_chunk = ts_fchunks_tuple
        fm_chunk = jfft.fftshift(jfft.fft(
            f_chunk, #[:nphis[t]], #[~np.isnan(f_chunk)], #f[index : index + nphi], ---how to get only not nan?
            norm="backward")) 
        ftm = ftm.at[t].add(spectral_periodic_extension_jax(fm_chunk, L, jnp))
        return ftm, None

    ftm,_ = jax.lax.scan(
        accumulate,
        jnp.zeros(
            samples.ftm_shape(L, "healpix", nside),
            dtype=jnp.complex128), #carry
        (ts,f_chunks_padded)) # array/PyTree of arrays to scan over 
    return ftm

########################
def healpix_fft_jax_3(f: np.ndarray, L: int, nside: int, numpy_module=np) -> np.ndarray:
    '''
    Healpix FFT JAX implementation using jax.numpy

    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    '''

    assert L >= 2 * nside
    ntheta = samples.ntheta(L, "healpix", nside)
    index = 0
    ftm_rows = []
    for t in range(ntheta):
        nphi = samples.nphi_ring(t, nside)
        fm_chunk = numpy_module.fft.fftshift(
            numpy_module.fft.fft(f[index : index + nphi], norm="backward")
        )
        ftm_rows.append(spectral_periodic_extension_jax(fm_chunk, L, numpy_module))
        index += nphi
    return numpy_module.stack(ftm_rows)

##########################
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
