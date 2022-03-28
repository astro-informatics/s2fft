import numpy as np
import os
import pyssht as ssht

import s2fft.logs as lg
lg.setup_logging()


def construct_ssht_legendre_matrix(
    L=4, sampling_method="MW", save_dir="../../.matrices"
):
    """Constructs associated Legendre matrix for precompute method

    Args:

        L (int): Angular bandlimit
        sampling_method (str): Sampling method to consider
        save_dir (str): Directory in which to save precomputed matrices

    Returns:

        Associated legendre matrix for forward ssht transform
    """

    Reality = False

    input_shape = ssht.sample_shape(L, Method=sampling_method)

    lg.info_log(
        "Ssht sampling {} selected with angular bandlimit {}".format(sampling_method, L)
    )

    Legendre = np.zeros((L * L, input_shape[0]), dtype=np.complex128)

    for i in range(input_shape[0]):

        lg.debug_log("Starting computing for theta index = {}".format(i))

        # Initialize
        in_matrix = np.zeros(input_shape, dtype=np.complex128)
        in_matrix[i, 0] = 1.0
        # Compute eigenvector and Reshape into a C major 1D column vector of length input_length
        Legendre[:, i] = ssht.forward(
            f=in_matrix, L=L, Method=sampling_method, Reality=Reality, Spin=0
        ).flatten("C")

        lg.debug_log("Ending computing for theta index = {}".format(i))

    Legendre_reshaped = np.zeros((L, 2 * L - 1, input_shape[0]), dtype=np.complex128)

    for l in range(L):
        for m in range(-l, l + 1):
            if m < 0:
                ind_tf = Legendre_reshaped.shape[1] + m
            if m >= 0:
                ind_tf = m
            ind_ssht = ssht.elm2ind(l, m)
            Legendre_reshaped[l, ind_tf, :] = Legendre[ind_ssht, :]
    Legendre = Legendre_reshaped

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}ssht_legendre_matrix_{}_{}_spin_0".format(
            save_dir, L, sampling_method
        )
    else:
        filename = "ssht_legendre_matrix_{}_{}_spin_0".format(L, sampling_method)

    lg.debug_log("Saving matrix binary to {}".format(filename))

    np.save(filename, Legendre)
    return Legendre_reshaped


def construct_ssht_legendre_matrix_inverse(
    L=4, sampling_method="MW", save_dir="../../.matrices"
):
    """Constructs associated Legendre inverse matrix for precompute method

    Args:

        L (int): Angular bandlimit
        sampling_method (str): Sampling method to consider
        save_dir (str): Directory in which to save precomputed matrices

    Returns:

        Associated legendre matrix for forward ssht transform
    """

    compile_warnings(L)

    Reality = False

    input_shape = ssht.sample_shape(L, Method=sampling_method)

    lg.info_log(
        "Ssht sampling {} selected with angular bandlimit {}".format(sampling_method, L)
    )

    Legendre_inverse = np.zeros((L * L, input_shape[0]), dtype=np.complex128)
    alm = np.zeros(L * L, dtype=np.complex128)

    for l in range(L):

        lg.debug_log("Starting computing for l = {}".format(l))

        for m in range(-l, l + 1):
            ind = ssht.elm2ind(l, m)
            alm[:] = 0.0
            alm[ind] = 1.0
            Legendre_inverse[ind, :] = ssht.inverse(
                flm=alm, L=L, Method=sampling_method, Reality=Reality, Spin=0
            )[:, 0]

        lg.debug_log("Ending computing for l = {}".format(l))

    Legendre_reshaped = np.zeros((L, 2 * L - 1, input_shape[0]), dtype=np.complex128)

    for l in range(L):
        for m in range(-l, l + 1):
            if m < 0:
                ind_tf = Legendre_reshaped.shape[1] + m
            if m >= 0:
                ind_tf = m
            ind_ssht = ssht.elm2ind(l, m)
            Legendre_reshaped[l, ind_tf, :] = Legendre_inverse[ind_ssht, :]
    Legendre_inverse = Legendre_reshaped

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}ssht_legendre_inverse_matrix_{}_{}_spin_0".format(
            save_dir, L, sampling_method
        )
    else:
        filename = "ssht_legendre_inverse_matrix_{}_{}_spin_0".format(
            L, sampling_method
        )

    lg.debug_log("Saving matrix binary to {}".format(filename))

    np.save(filename, Legendre_inverse)

    return Legendre_inverse


def compile_warnings(L):
    base_value = 10
    if L >= 128:
        lg.warning_log(
            "Inverse associated Legendre matrix precomputation currently scales as L^5 -- get a coffee this may take a while."
        )
        lg.debug_log(
            "L = {} matrix compilation takes ~{} hours".format(
                L, (base_value * (L / 128) ** 2) / 60.0
            )
        )


if __name__ == "__main__":
    L = 32
    N = 1

    m1 = construct_ssht_legendre_matrix(L=L)
    m2 = construct_ssht_legendre_matrix_inverse(L=L)

    print(
        "Forward Legendre Matrix: Total entries = {}, Sparse entries = {}".format(
            len(m1.flatten("C")), np.count_nonzero(m1)
        )
    )
    print(
        "Inverse Legendre Matrix: Total entries = {}, Sparse entries = {}".format(
            len(m2.flatten("C")), np.count_nonzero(m2)
        )
    )
