import jax.numpy as jnp
import argparse


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


def launch_spectral_extension(nside, fm: jnp.ndarray, L: int) -> jnp.ndarray:
    start_index = 0
    end_index = total_pixels

    nphi = 0
    upper_rings_offset = []
    lower_rings_offset = []

    list_of_spectrals = []

    for i in range(nside - 1):
        nphi = 4 * (i + 1)
        upper_rings_offset.append((start_index, start_index + nphi))
        lower_rings_offset.append((end_index - nphi, end_index))

        start_index += nphi
        end_index -= nphi

    equator_offset = start_index
    equator_size = 4 * nside
    equator_ring_number = (end_index - start_index) // (4 * nside)

    for upper_ring in upper_rings_offset:
        start_offset, end_offset = upper_ring
        nphi = end_offset - start_offset
        print("start_offset: ", start_offset, "end_offset: ", end_offset,
              "Size of array: ", nphi)
        k_a = fm[start_offset:end_offset]
        list_of_spectrals.append(spectral_periodic_extension_jax(k_a, L))

    for equator_ring in range(equator_ring_number):
        start_offset = equator_offset + equator_ring * equator_size
        end_offset = start_offset + equator_size
        nphi = end_offset - start_offset
        print("start_offset: ", start_offset, "end_offset: ", end_offset,
              "Size of array: ", nphi)
        k_a = fm[start_offset:end_offset]
        list_of_spectrals.append(spectral_periodic_extension_jax(k_a, L))

    for lower_ring in lower_rings_offset[::-1]:
        start_offset, end_offset = lower_ring
        nphi = end_offset - start_offset
        print("start_offset: ", start_offset, "end_offset: ", end_offset,
              "Size of array: ", nphi)
        k_a = fm[start_offset:end_offset]
        list_of_spectrals.append(spectral_periodic_extension_jax(k_a, L))

    return jnp.concatenate(list_of_spectrals)


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


def launch_spectral_folding(nside, fm: jnp.ndarray, L: int) -> jnp.ndarray:
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Healpix FFT')
    parser.add_argument('-s',
                        '--nside',
                        type=int,
                        help='Healpix nside',
                        default=4)
    parser.add_argument('-L', '--L', type=int, help='Healpix lmax', default=8)
    parser.add_argument('-t',
                        '--type',
                        type=str,
                        help="spectrial type : extended, folded, both",
                        default='both')
    parser.add_argument('-c',
                        '--check',
                        action='store_true',
                        help='Check result [Boolean]')
    parser.add_argument('-p',
                        '--print',
                        help='Print results [Boolean]',
                        action='store_true')
    args = parser.parse_args()

    assert args.type in ['extended', 'folded', 'both'], "Invalid type"
    assert args.L >= 2 * args.nside, "L must be greater than or equal to 2 * nside"

    total_pixels = 12 * args.nside**2

    healpix_array = jnp.arange(total_pixels)

    match args.type:
        case 'extended':
            result = launch_spectral_extension(args.nside, healpix_array,
                                               args.L)
        case 'folded':
            result = launch_spectral_folding(args.nside, healpix_array, args.L)
        case 'both':
            result = launch_spectral_extension(args.nside, healpix_array,
                                               args.L)
            result = launch_spectral_folding(args.nside, result, args.L)

    if args.print:
        for i, elem in enumerate(result):
            print(f"[{i}] {elem} ")
