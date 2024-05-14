import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import argparse
from jax import jit
from functools import partial


# enum class for FWD BWD both
class FFTType:
    FWD = 1
    BWD = 2
    BOTH = 3
    NONE = 4


def fshift(a, shift_bool):
    if shift_bool:
        return fftshift(a)
    return a


def ifshift(a, shift_bool):
    if shift_bool:
        return ifftshift(a)
    return a


def generate_healpix(nside):
    total_pixels = 12 * nside * nside
    return np.arange(total_pixels)


def healpix_fft(array, nside, total_pixels, shift, norm):

    start_index = 0
    end_index = total_pixels

    nphi = 0
    upper_rings_offset = []
    lower_rings_offset = []

    list_of_ffts = []

    for i in range(nside - 1):
        nphi = 4 * (i + 1)
        upper_rings_offset.append((start_index, start_index + nphi))
        lower_rings_offset.append((end_index - nphi, end_index))

        start_index += nphi
        end_index -= nphi

    equator_offset = start_index
    equator_size = 4 * nside
    equator_ring_number = (end_index - start_index) // (4 * nside)

    lower_rings_offset = lower_rings_offset[::-1]

    for i, upper_ring in enumerate(upper_rings_offset):
        start_offset, end_offset = upper_ring
        print(
            f"start_offset: {start_offset} end_offset: {end_offset} Size of array: {end_offset - start_offset} at {i}"
        )
        k_a = array[start_offset:end_offset]
        list_of_ffts.append(fshift(fft(k_a, norm=norm), shift))

    print(f"equator")

    for equator_ring in range(equator_ring_number):
        start_offset = equator_offset + equator_ring * equator_size
        end_offset = start_offset + equator_size
        print(
            f"start_offset: {start_offset} end_offset: {end_offset} Size of array: {end_offset - start_offset} at {(equator_ring + nside - 1)}"
        )
        k_a = array[start_offset:end_offset]
        #list_of_ffts.append(fshift(fft(k_a, norm=norm), shift))
        list_of_ffts.append(k_a)

    print(f"lower")

    for i, lower_ring in enumerate(lower_rings_offset):
        start_offset, end_offset = lower_ring
        print(
            f"start_offset: {start_offset} end_offset: {end_offset} Size of array: {end_offset - start_offset} at {(i + 3*nside)}"
        )
        k_a = array[start_offset:end_offset]
        list_of_ffts.append(fshift(fft(k_a, norm=norm), shift))

    k_array = np.concatenate(list_of_ffts)

    return k_array


def healpix_ifft(array, nside, total_pixels, shift, norm):

    start_index = 0
    end_index = total_pixels

    nphi = 0
    upper_rings_offset = []
    lower_rings_offset = []

    list_of_iffts = []

    for i in range(nside - 1):
        nphi = 4 * (i + 1)
        upper_rings_offset.append((start_index, start_index + nphi))
        lower_rings_offset.append((end_index - nphi, end_index))

        start_index += nphi
        end_index -= nphi

    equator_offset = start_index
    equator_size = 4 * nside
    equator_ring_number = (end_index - start_index) // (4 * nside)

    lower_rings_offset = lower_rings_offset[::-1]

    for upper_ring in upper_rings_offset:
        start_offset, end_offset = upper_ring
        print(f"start_offset: {start_offset} end_offset: {end_offset}")
        k_a = array[start_offset:end_offset]
        list_of_iffts.append(ifft(ifshift(k_a, shift), norm=norm))

    print(f"equator")

    for equator_ring in range(equator_ring_number):
        start_offset = equator_offset + equator_ring * equator_size
        print(f"start_offset: {start_offset} end_offset: {end_offset}")
        end_offset = start_offset + equator_size
        k_a = array[start_offset:end_offset]
        #list_of_iffts.append(ifft(ifshift(k_a, shift), norm=norm))
        list_of_iffts.append(k_a)

    print(f"lower")

    for lower_ring in lower_rings_offset:
        start_offset, end_offset = lower_ring
        print(f"start_offset: {start_offset} end_offset: {end_offset}")
        k_a = array[start_offset:end_offset]
        list_of_iffts.append(ifft(ifshift(k_a, shift), norm=norm))

    k_array = np.concatenate(list_of_iffts)

    return k_array


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Healpix FFT')
    parser.add_argument('-s',
                        '--nside',
                        type=int,
                        help='Healpix nside',
                        default=4)
    parser.add_argument('-t',
                        '--ffttype',
                        type=str,
                        help="FFTType : fwd bwd both",
                        default='both')
    parser.add_argument('-sh',
                        '--shift',
                        action='store_true',
                        help="fftshift result [Boolean]")
    parser.add_argument('-n',
                        '--norm',
                        type=str,
                        help='Normalization : bwd fwd ortho',
                        default='bwd')
    parser.add_argument('-c',
                        '--check',
                        action='store_true',
                        help='Check result [Boolean]')
    parser.add_argument('-p',
                        '--print',
                        help='Print results [Boolean]',
                        action='store_true')
    args = parser.parse_args()

    shift_bool = args.shift
    fft_type = FFTType.NONE

    match args.ffttype:
        case 'fwd':
            fft_type = FFTType.FWD
        case 'bwd':
            fft_type = FFTType.BWD
        case _:
            fft_type = FFTType.BOTH

    norm = 'backward'
    match args.norm:
        case 'bwd':
            norm = 'backward'
        case 'fwd':
            norm = 'forward'
        case 'ortho':
            norm = 'ortho'
        case _:
            norm = 'backward'

    a = generate_healpix(args.nside)
    total_pixels = 12 * args.nside * args.nside
    nside = args.nside

    match fft_type:
        case FFTType.FWD:
            result = healpix_fft(a,
                                 nside,
                                 total_pixels,
                                 shift=shift_bool,
                                 norm=norm)
        case FFTType.BWD:
            result = healpix_ifft(a,
                                  nside,
                                  total_pixels,
                                  shift=shift_bool,
                                  norm=norm)
        case FFTType.BOTH:
            k_array = healpix_fft(a,
                                  nside,
                                  total_pixels,
                                  shift=shift_bool,
                                  norm=norm)
            result = healpix_ifft(k_array,
                                  nside,
                                  total_pixels,
                                  shift=shift_bool,
                                  norm=norm)
            if args.check:
                print(f"max error: {np.max(np.abs(result - a))}")
        case FFTType.NONE:
            print("No FFT type selected")

    if args.print:
        for i, elem in enumerate(result):
            print(f"[{i}] {elem.real} + {elem.imag}j")
