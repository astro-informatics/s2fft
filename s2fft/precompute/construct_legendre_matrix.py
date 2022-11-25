import numpy as np
import os
import pyssht as ssht

from s2fft.spherical import samples
import s2fft.logs as lg

lg.setup_logging()


def construct_legendre_matrix(
    L: int = 4,
    sampling_method: str = "mw",
    save_dir: str = "../../.matrices",
    spin: int = 0,
):
    """Construct associated Legendre matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.

        sampling_method (str, optional): Sampling method to consider (Currently "MW" only supported).

        save_dir (str, optional): Directory in which to save precomputed matrices.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Associated Legendre matrix for forward harmonic transform.
    """

    ntheta = samples.ntheta(L, sampling_method)
    nphi = samples.nphi_equiang(L, sampling_method)

    lg.info_log(
        "Sampling {} selected with angular bandlimit {} and spin {}".format(
            sampling_method, L, spin
        )
    )

    Legendre = np.zeros((L * L, ntheta), dtype=np.complex128)

    for i in range(ntheta):

        lg.debug_log("Starting computing for theta index = {}".format(i))

        in_matrix = np.zeros((ntheta, nphi), dtype=np.complex128)
        in_matrix[i, 0] = 1.0

        Legendre[:, i] = ssht.forward(
            f=in_matrix, L=L, Method=sampling_method.upper(), Spin=spin
        ).flatten("C")

        lg.debug_log("Ending computing for theta index = {}".format(i))

    Legendre_reshaped = np.zeros((L, 2 * L - 1, ntheta), dtype=np.complex128)

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
        filename = "{}legendre_matrix_{}_{}_spin_{}".format(
            save_dir, L, sampling_method, spin
        )
    else:
        filename = "legendre_matrix_{}_{}_spin_{}".format(L, sampling_method, spin)

    lg.debug_log("Saving matrix binary to {}".format(filename))

    np.save(filename, Legendre)
    return Legendre_reshaped


def construct_legendre_matrix_inverse(
    L: int = 4,
    sampling_method: str = "mw",
    save_dir: str = "../../.matrices",
    spin: int = 0,
):
    """Construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.

        sampling_method (str, optional): Sampling method to consider (Currently "MW" only supported).

        save_dir (str, optional): Directory in which to save precomputed matrices.

        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Associated Legendre matrix for inverse harmonic transform.
    """

    compute_warning(L)

    ntheta = samples.ntheta(L, sampling_method)

    lg.info_log(
        "Sampling {} selected with angular bandlimit {} and spin {}".format(
            sampling_method, L, spin
        )
    )

    Legendre_inverse = np.zeros((L * L, ntheta), dtype=np.complex128)
    alm = np.zeros(L * L, dtype=np.complex128)

    for l in range(L):

        lg.debug_log("Starting computing for l = {}".format(l))

        for m in range(-l, l + 1):
            ind = ssht.elm2ind(l, m)
            alm[:] = 0.0
            alm[ind] = 1.0
            Legendre_inverse[ind, :] = ssht.inverse(
                flm=alm, L=L, Method=sampling_method.upper(), Spin=spin
            )[:, 0]

        lg.debug_log("Ending computing for l = {}".format(l))

    Legendre_reshaped = np.zeros((L, 2 * L - 1, ntheta), dtype=np.complex128)

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
        filename = "{}legendre_inverse_matrix_{}_{}_spin_{}".format(
            save_dir, L, sampling_method, spin
        )
    else:
        filename = "legendre_inverse_matrix_{}_{}_spin_{}".format(
            L, sampling_method, spin
        )

    lg.debug_log("Saving matrix binary to {}".format(filename))

    np.save(filename, Legendre_inverse)

    return Legendre_inverse


def compute_warning(L):
    r"""Basic compute warning for large Legendre precomputes.

    Args:
        L (int): Harmonic band-limit.

    Raises:
        Warning: If the estimated time for precompute is large (:math:`L>128`).
    """

    base_value = 10
    if L >= 256:
        lg.warning_log(
            "Inverse associated Legendre matrix precomputation currently scales as L^5 -- get a coffee this may take a while."
        )
        lg.debug_log(
            "L = {} matrix computation takes ~{} hours".format(
                L, (base_value * (L / 128) ** 2) / 60.0
            )
        )


def load_legendre_matrix(
    L=4, sampling_method="mw", save_dir="../../.matrices", forward=True, spin=0
):
    """Construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.

        sampling_method (str, optional): Sampling method to consider (Currently "MW" only supported).

        save_dir (str, optional): Directory from which to load precomputed matrices.

        forward (bool, optional): Whether to load the forward or inverse matrices.

        spin (int, optional): Spin of the transform to consider.

    Returns:
        np.ndarray: Associated Legendre matrix for corresponding harmonic transform.
    """

    dir_string = ""
    if not forward:
        dir_string += "_inverse"

    filepath = "{}/legendre{}_matrix_{}_{}_spin_{}.npy".format(
        save_dir, dir_string, L, sampling_method, spin
    )

    if not os.path.isfile(filepath):
        if forward:
            construct_legendre_matrix(
                L=L,
                sampling_method=sampling_method,
                save_dir=save_dir,
                spin=spin,
            )
        else:
            construct_legendre_matrix_inverse(
                L=L,
                sampling_method=sampling_method,
                save_dir=save_dir,
                spin=spin,
            )
    return np.load(filepath)
