from jax import jit, vmap

import numpy as np
import jax.numpy as jnp
import torch
from s2fft.sampling import s2_samples as samples
from functools import partial
from s2fft.utils.jax_pritimive import BasePrimitive, register_primitive
import jaxlib.mlir.ir as ir
from s2fft_lib import _s2fft
from jaxlib.hlo_helpers import custom_call
from jax.core import  ShapedArray
from typing import Tuple
from jax import ShapeDtypeStruct
# did not find promote_dtypes_complex outside _src
from jax._src.numpy.util import promote_dtypes_complex


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


def spectral_folding_jax(fm: jnp.ndarray, nphi: int, L: int) -> jnp.ndarray:
  """Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. aliasing high frequencies. JAX specific implementation of
    :func:`~spectral_folding`.

    Args:
        fm (jnp.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Lower resolution set of aliased Fourier coefficients.
    """
  slice_start = L - nphi // 2
  slice_stop = slice_start + nphi
  ftm_slice = fm[slice_start:slice_stop]

  ftm_slice = ftm_slice.at[-jnp.arange(1, L - nphi // 2 + 1) % nphi].add(
      fm[slice_start - jnp.arange(1, L - nphi // 2 + 1)])
  return ftm_slice.at[jnp.arange(L - nphi // 2) % nphi].add(
      fm[slice_stop + jnp.arange(L - nphi // 2)])


def spectral_folding_torch(fm: torch.tensor, nphi: int, L: int) -> torch.tensor:
  """Folds higher frequency Fourier coefficients back onto lower frequency
    coefficients, i.e. aliasing high frequencies. Torch specific implementation of
    :func:`~spectral_folding`.

    Args:
        fm (torch.tensor): Slice of Fourier coefficients corresponding to ring at latitute t.

        nphi (int): Total number of pixel space phi samples for latitude t.

        L (int): Harmonic band-limit.

    Returns:
        torch.tensor: Lower resolution set of aliased Fourier coefficients.
    """
  slice_start = L - nphi // 2
  slice_stop = slice_start + nphi
  ftm_slice = fm[slice_start:slice_stop]

  ftm_slice = ftm_slice.put_(
      -torch.arange(1, L - nphi // 2 + 1) % nphi,
      fm[slice_start - torch.arange(1, L - nphi // 2 + 1)],
      accumulate=True,
  )
  ftm_slice = ftm_slice.put_(
      torch.arange(L - nphi // 2) % nphi,
      fm[slice_stop + torch.arange(L - nphi // 2)],
      accumulate=True,
  )

  return ftm_slice


def spectral_periodic_extension(fm: np.ndarray, nphi: int,
                                L: int) -> np.ndarray:
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


def spectral_periodic_extension_jax(fm: jnp.ndarray, L: int) -> jnp.ndarray:
  """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on
    :func:`~spectral_periodic_extension`, modified to be JIT-compilable.

    Args:
        fm (jnp.ndarray): Slice of Fourier coefficients corresponding to ring at latitute t.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Higher resolution set of periodic Fourier coefficients.
    """
  nphi = fm.shape[0]
  return jnp.concatenate((
      fm[-jnp.arange(L - nphi // 2, 0, -1) % nphi],
      fm,
      fm[jnp.arange(L - (nphi + 1) // 2) % nphi],
  ))


def spectral_periodic_extension_torch(fm: torch.tensor, L: int) -> torch.tensor:
  """Extends lower frequency Fourier coefficients onto higher frequency
    coefficients, i.e. imposed periodicity in Fourier space. Based on
    :func:`~spectral_periodic_extension`.

    Args:
        fm (torch.tensor): Slice of Fourier coefficients corresponding to ring at latitute t.

        L (int): Harmonic band-limit.

    Returns:
        torch.tensor: Higher resolution set of periodic Fourier coefficients.
    """
  nphi = fm.shape[0]
  return torch.concatenate((
      fm[-torch.arange(L - nphi // 2, 0, -1) % nphi],
      fm,
      fm[torch.arange(L - (nphi + 1) // 2) % nphi],
  ))


def healpix_fft(
    f: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
  """Wrapper function for the Forward Fast Fourier Transform with spectral
    back-projection in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax", "torch"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax", "torch"}.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
  if method.lower() == "numpy":
    return healpix_fft_numpy(f, L, nside, reality)
  elif method.lower() == "jax":
    return healpix_fft_jax(f, L, nside, reality)
  elif method.lower() == "cuda":
    return healpix_fft_cuda(f, L, nside, reality)
  elif method.lower() == "torch":
    return healpix_fft_torch(f, L, nside, reality)
  else:
    raise ValueError(f"Method {method} not recognised.")


def healpix_fft_numpy(f: np.ndarray, L: int, nside: int,
                      reality: bool) -> np.ndarray:
  """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity.

    Args:
        f (np.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        np.ndarray: Array of Fourier coefficients for all latitudes.
    """
  index = 0
  ftm = np.zeros(samples.ftm_shape(L, "healpix", nside), dtype=np.complex128)
  ntheta = ftm.shape[0]
  for t in range(ntheta):
    nphi = samples.nphi_ring(t, nside)
    if reality and nphi == 2 * L:
      fm_chunk = np.zeros(nphi, dtype=np.complex128)
      fm_chunk[nphi // 2:] = np.fft.rfft(
          np.real(f[index:index + nphi]), norm="backward")[:-1]
    else:
      fm_chunk = np.fft.fftshift(
          np.fft.fft(f[index:index + nphi], norm="backward"))
    ftm[t] = (
        fm_chunk if nphi == 2 *
        L else spectral_periodic_extension(fm_chunk, nphi, L))
    index += nphi
  return ftm


@partial(jit, static_argnums=(1, 2, 3))
def healpix_fft_jax(f: jnp.ndarray, L: int, nside: int,
                    reality: bool) -> jnp.ndarray:
  """
    Healpix FFT JAX implementation using jax.numpy/numpy stack
    Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity. JAX specific
    implementation of :func:`~healpix_fft_numpy`.

    Args:
        f (jnp.ndarray): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: Array of Fourier coefficients for all latitudes.
    """

  def f_chunks_to_ftm_rows(f_chunks, nphi):
    if reality and nphi == 2 * L:
      fm_chunks = jnp.concatenate(
          (
              jnp.zeros((f_chunks.shape[0], nphi // 2)),
              jnp.fft.rfft(jnp.real(f_chunks), norm="backward")[:, :-1],
          ),
          axis=1,
      )
    else:
      fm_chunks = jnp.fft.fftshift(
          jnp.fft.fft(f_chunks, norm="backward"), axes=-1)
    return vmap(spectral_periodic_extension_jax, (0, None))(fm_chunks, L)

  # Process f chunks corresponding to pairs of polar theta rings with the same number
  # of phi samples together to reduce size of unrolled traced computational graph
  ftm_rows_polar = []
  start_index, end_index = 0, 12 * nside**2
  for t in range(0, nside - 1):
    nphi = 4 * (t + 1)
    f_chunks = jnp.stack(
        (f[start_index:start_index + nphi], f[end_index - nphi:end_index]))
    ftm_rows_polar.append(f_chunks_to_ftm_rows(f_chunks, nphi))
    start_index, end_index = start_index + nphi, end_index - nphi
  ftm_rows_polar = jnp.stack(ftm_rows_polar)
  # Process all f chunks for the equal sized equatorial theta rings together
  nphi = 4 * nside
  f_chunks_equatorial = f[start_index:end_index].reshape((-1, nphi))
  ftm_rows_equatorial = f_chunks_to_ftm_rows(f_chunks_equatorial, nphi)
  # Concatenate Fourier coefficients for all latitudes, reversing second polar set to
  # account for processing order
  return jnp.concatenate((
      ftm_rows_polar[:, 0],
      ftm_rows_equatorial,
      ftm_rows_polar[::-1, 1],
  ))


def healpix_fft_torch(f: torch.tensor, L: int, nside: int,
                      reality: bool) -> torch.tensor:
  """Computes the Forward Fast Fourier Transform with spectral back-projection
    in the polar regions to manually enforce Fourier periodicity. Torch specific
    implementation of :func:`~healpix_fft_numpy`.

    Args:
        f (torch.tensor): HEALPix pixel-space array.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        torch.tensor: Array of Fourier coefficients for all latitudes.
    """
  index = 0
  ftm = torch.zeros(
      samples.ftm_shape(L, "healpix", nside), dtype=torch.complex128)
  ntheta = ftm.shape[0]
  for t in range(ntheta):
    nphi = samples.nphi_ring(t, nside)
    if reality and nphi == 2 * L:
      fm_chunk = torch.zeros(nphi, dtype=torch.complex128)
      fm_chunk[nphi // 2:] = torch.fft.rfft(
          torch.real(f[index:index + nphi]), norm="backward")[:-1]
    else:
      fm_chunk = torch.fft.fftshift(
          torch.fft.fft(f[index:index + nphi], norm="backward"))
    ftm[t] = (
        fm_chunk if nphi == 2 *
        L else spectral_periodic_extension_torch(fm_chunk, L))
    index += nphi
  return ftm


def healpix_ifft(
    ftm: np.ndarray,
    L: int,
    nside: int,
    method: str = "numpy",
    reality: bool = False,
) -> np.ndarray:
  """Wrapper function for the Inverse Fast Fourier Transform with spectral folding
    in the polar regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        method (str, optional): Evaluation method in {"numpy", "jax", "torch"}.
            Defaults to "numpy".

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax", "torch"}.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
  assert L >= 2 * nside
  if method.lower() == "numpy":
    return healpix_ifft_numpy(ftm, L, nside, reality)
  elif method.lower() == "jax":
    return healpix_ifft_jax(ftm, L, nside, reality)
  elif method.lower() == "cuda":
    return healpix_ifft_cuda(ftm, L, nside, reality)
  elif method.lower() == "torch":
    return healpix_ifft_torch(ftm, L, nside, reality)
  else:
    raise ValueError(f"Method {method} not recognised.")


def healpix_ifft_numpy(ftm: np.ndarray, L: int, nside: int,
                       reality: bool) -> np.ndarray:
  """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing.

    Args:
        ftm (np.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        np.ndarray: HEALPix pixel-space array.
    """
  f = np.zeros(
      samples.f_shape(sampling="healpix", nside=nside), dtype=np.complex128)
  ntheta = ftm.shape[0]
  index = 0
  for t in range(ntheta):
    nphi = samples.nphi_ring(t, nside)
    fm_chunk = ftm[t] if nphi == 2 * L else spectral_folding(ftm[t], nphi, L)
    if reality and nphi == 2 * L:
      f[index:index + nphi] = np.fft.irfft(
          fm_chunk[nphi // 2:], nphi, norm="forward")
    else:
      f[index:index + nphi] = np.fft.ifft(
          np.fft.ifftshift(fm_chunk), norm="forward")
    index += nphi
  return f


@partial(jit, static_argnums=(1, 2, 3))
def healpix_ifft_jax(ftm: jnp.ndarray, L: int, nside: int,
                     reality: bool) -> jnp.ndarray:
  """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing, using JAX. JAX specific implementation of
    :func:`~healpix_ifft_numpy`.

    Args:
        ftm (jnp.ndarray): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        jnp.ndarray: HEALPix pixel-space array.
    """

  def ftm_rows_to_f_chunks(ftm_rows, nphi):
    fm_chunks = (
        ftm_rows if nphi == 2 * L else vmap(spectral_folding_jax,
                                            (0, None, None))(ftm_rows, nphi, L))
    if reality and nphi == 2 * L:
      return jnp.fft.irfft(fm_chunks[:, nphi // 2:], nphi, norm="forward")
    else:
      return jnp.fft.ifft(
          jnp.fft.ifftshift(fm_chunks, axes=-1), norm="backward")

  # Process ftm rows corresponding to pairs of polar theta rings with the same number
  # of phi samples together to reduce size of unrolled traced computational graph
  f_chunks_polar = [
      ftm_rows_to_f_chunks(jnp.stack((ftm[t], ftm[-(t + 1)])), 4 * (t + 1))
      for t in range(nside - 1)
  ]
  # Process all ftm rows for the equal sized equatorial theta rings together
  f_chunks_equatorial = ftm_rows_to_f_chunks(ftm[nside - 1:3 * nside],
                                             4 * nside)
  # Concatenate f chunks for all theta rings together, reversing second polar set
  # to account for processing order
  return jnp.concatenate(
      [f_chunks_polar[t][0] for t in range(nside - 1)] +
      [f_chunks_equatorial.flatten()] +
      [f_chunks_polar[t][1] for t in reversed(range(nside - 1))])


def healpix_ifft_torch(ftm: torch.tensor, L: int, nside: int,
                       reality: bool) -> torch.tensor:
  """Computes the Inverse Fast Fourier Transform with spectral folding in the polar
    regions to mitigate aliasing. Torch specific implementation of
    :func:`~healpix_ifft_numpy`.

    Args:
        ftm (torch.tensor): Array of Fourier coefficients for all latitudes.

        L (int): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        reality (bool): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

    Returns:
        torch.tensor: HEALPix pixel-space array.
    """
  f = torch.zeros(
      samples.f_shape(sampling="healpix", nside=nside), dtype=torch.complex128)
  ntheta = ftm.shape[0]
  index = 0
  for t in range(ntheta):
    nphi = samples.nphi_ring(t, nside)
    fm_chunk = ftm[t] if nphi == 2 * L else spectral_folding_torch(
        ftm[t], nphi, L)
    if reality and nphi == 2 * L:
      f[index:index + nphi] = torch.fft.irfft(
          fm_chunk[nphi // 2:], nphi, norm="forward")
    else:
      f[index:index + nphi] = torch.fft.ifft(
          torch.fft.ifftshift(fm_chunk), norm="forward")
    index += nphi
  return f


def p2phi_rings(t: np.ndarray, nside: int) -> np.ndarray:
  r"""Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
    Vectorised implementation of :func:`~samples.p2phi_ring`.

    Args:
        t (np.ndarrray): vector of :math:`\theta` ring indicies, i.e. [0,1,...,ntheta-1]

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        np.ndarray: :math:`\phi` offset for each ring index.
    """
  shift = 1 / 2
  tt = np.zeros_like(t)
  tt = np.where(
      (t + 1 >= nside) & (t + 1 <= 3 * nside),
      shift * ((t - nside + 2) % 2) * np.pi / (2 * nside),
      tt,
  )
  tt = np.where(t + 1 > 3 * nside, shift * np.pi / (2 * (4 * nside - t - 1)),
                tt)
  tt = np.where(t + 1 < nside, shift * np.pi / (2 * (t + 1)), tt)
  return tt


@partial(jit, static_argnums=(1))
def p2phi_rings_jax(t: jnp.ndarray, nside: int) -> jnp.ndarray:
  r"""Convert index to :math:`\phi` angle for HEALPix for all :math:`\theta` rings.
    JAX implementation of :func:`~p2phi_rings`.

    Args:
        t (jnp.ndarrray): vector of :math:`\theta` ring indicies, i.e. [0,1,...,ntheta-1]

        nside (int): HEALPix Nside resolution parameter.

    Returns:
        jnp.ndarray: :math:`\phi` offset for each ring index.
    """
  shift = 1 / 2
  tt = jnp.zeros_like(t)
  tt = jnp.where(
      (t + 1 >= nside) & (t + 1 <= 3 * nside),
      shift * ((t - nside + 2) % 2) * jnp.pi / (2 * nside),
      tt,
  )
  tt = jnp.where(t + 1 > 3 * nside, shift * jnp.pi / (2 * (4 * nside - t - 1)),
                 tt)
  tt = jnp.where(t + 1 < nside, shift * jnp.pi / (2 * (t + 1)), tt)
  return tt


def ring_phase_shifts_hp(L: int,
                         nside: int,
                         forward: bool = False,
                         reality: bool = False) -> np.ndarray:
  r"""Generates a phase shift vector for HEALPix for all :math:`\theta` rings.

    Args:
        L (int, optional): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Returns:
        np.ndarray: Vector of phase shifts with shape :math:`[n_{\theta}, 2L-1]`.
    """
  t = np.arange(samples.ntheta(L, "healpix", nside))
  phi_offsets = p2phi_rings(t, nside)
  sign = -1 if forward else 1
  m_start_ind = 0 if reality else -L + 1
  exponent = np.einsum("t, m->tm", phi_offsets, np.arange(m_start_ind, L))
  return np.exp(sign * 1j * exponent)


@partial(jit, static_argnums=(0, 1, 2, 3))
def ring_phase_shifts_hp_jax(L: int,
                             nside: int,
                             forward: bool = False,
                             reality: bool = False) -> jnp.ndarray:
  r"""Generates a phase shift vector for HEALPix for all :math:`\theta` rings. JAX
    implementation of :func:`~ring_phase_shifts_hp`.

    Args:
        L (int, optional): Harmonic band-limit.

        nside (int): HEALPix Nside resolution parameter.

        forward (bool, optional): Whether to provide forward or inverse shift.
            Defaults to False.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

    Returns:
        jnp.ndarray: Vector of phase shifts with shape :math:`[n_{\theta}, 2L-1]`.
    """
  t = jnp.arange(samples.ntheta(L, "healpix", nside))
  phi_offsets = p2phi_rings_jax(t, nside)
  sign = -1 if forward else 1
  m_start_ind = 0 if reality else -L + 1
  exponent = jnp.einsum(
      "t, m->tm", phi_offsets, jnp.arange(m_start_ind, L), optimize=True)
  return jnp.exp(sign * 1j * exponent)


## Cuda primitive ##


class HealpixFFTPrimitive(BasePrimitive):

  name = "HealpixFFT"
  multiple_results = False
  impl_static_args = (1, 2, 3, 4)
  inner_primitive = None
  outer_primitive = None

  @staticmethod
  def abstract(f, L, nside, reality, fft_type):

    # For the forward pass, the input is a HEALPix pixel-space array of size nside^2 * 12 and the output is
    # a FTM array of shape (number of rings , width of FTM slice) which is (4 * nside - 1 , 2 * L  )
    healpix_size = (nside**2 * 12,)
    ftm_size = (4 * nside - 1, 2 * L)
    if fft_type == 'forward':
      assert f.shape == healpix_size
      return f.update(shape=ftm_size, dtype=f.dtype)
    elif fft_type == 'backward':
      print(f"f.shape {f.shape}")
      assert f.shape == ftm_size
      return f.update(shape=healpix_size, dtype=f.dtype)
    else:
      raise ValueError(f"fft_type {fft_type} not recognised.")

  @staticmethod
  def lowering(ctx, f, *, L, nside, reality, fft_type):
    (x_aval,) = ctx.avals_in
    (aval_out,) = ctx.avals_out
    dtype = x_aval.dtype
    a_type = ir.RankedTensorType(f.type)

    out_dtype = aval_out.dtype
    if out_dtype == np.complex64:
      out_type = ir.ComplexType.get(ir.F32Type.get())
      is_double = False
    elif out_dtype == np.complex128:
      out_type = ir.ComplexType.get(ir.F64Type.get())
      is_double = True
    else:
      raise ValueError(f'Unknown output type {out_dtype}')

    out_type = ir.RankedTensorType.get(aval_out.shape, out_type)

    forward = fft_type == 'forward'

    # std::pair<size_t, nb::bytes> build_healpix_fft_descriptor(int nside, int harmonic_band_limit, bool reality,
    #                                                   bool forward, bool double_precision)
    opaque = _s2fft.build_healpix_fft_descriptor(nside, L, reality, forward,
                                                 is_double)

    layout = tuple(range(len(a_type.shape) - 1, -1, -1))
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))

    result = custom_call(
        "healpix_fft_cuda",
        result_types=[out_type],
        operands=[f],
        operand_layouts=[layout],
        result_layouts=[out_layout],
        has_side_effect=True,
        # For the future, this will be used in multi GPU healpix fft
        # operand_output_aliases={0: 0},
        backend_config=opaque,
    )
    # For multi GPU healpix fft, I will be using XLA buffer donation functionality
    # Which will return the same shape as the input in the CUDA primitive then it will be reshaped to the output shape
    # return hlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results
    return result.results

  @staticmethod
  def impl(f, L, nside, reality, fft_type):
    return HealpixFFTPrimitive.inner_primitive.bind(f, L, nside, reality,
                                                    fft_type)

  # Multi GPU part

  # @staticmethod
  # def per_shard_impl():
  #   return NotImplemented
  #
  # @staticmethod
  # def infer_sharding_from_operands(L, nside, reality, fft_type, mesh: Mesh,
  #                                  arg_infos: Tuple[ShapeDtypeStruct],
  #                                  result_infos: Tuple[ShapedArray]):
  #   return NotImplemented
  #
  # @staticmethod
  # def partition(L, nside, reality, fft_type, mesh: Mesh,
  #               arg_shapes: Tuple[Tuple[int]], result_shape: Tuple[int]):
  #   return NotImplemented
  #

register_primitive(HealpixFFTPrimitive)


@partial(jit, static_argnums=(1, 2, 3))
def healpix_fft_cuda(f, L, nside, reality):
  (f,) = promote_dtypes_complex(f)
  return HealpixFFTPrimitive.inner_primitive.bind(
      f, L=L, nside=nside, reality=reality, fft_type='forward')


@partial(jit, static_argnums=(1, 2, 3))
def healpix_ifft_cuda(f, L, nside, reality):
  (f,) = promote_dtypes_complex(f)
  return HealpixFFTPrimitive.inner_primitive.bind(
      f, L=L, nside=nside, reality=reality, fft_type='backward')
