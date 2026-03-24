from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeAlias

import numpy as np
import pytest

P = ParamSpec("P")
TestData: TypeAlias = Mapping[str, Any]


@pytest.fixture
def cached_so3_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flmn_generator: Callable[..., np.ndarray],
    _s2fft_to_so3_sampling: Callable[[str], str],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, N: int, L_lower: int, sampling: str, reality: bool
    ) -> dict[str, np.ndarray]:
        import so3

        from s2fft.sampling import so3_samples

        flmn = flmn_generator(L=L, N=N, L_lower=L_lower, reality=reality)

        so3_parameters = so3.create_parameter_dict(
            L=L,
            N=N,
            L0=L_lower,
            sampling_scheme_str=_s2fft_to_so3_sampling(sampling),
            reality=False,
        )

        f_so3 = so3.inverse(so3_samples.flmn_3d_to_1d(flmn, L, N), so3_parameters)
        flmn_so3 = so3_samples.flmn_1d_to_3d(so3.forward(f_so3, so3_parameters), L, N)

        return {"flmn": flmn, "f_so3": f_so3, "flmn_so3": flmn_so3}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.fixture
def cached_so3_samples_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    _s2fft_to_so3_sampling: Callable[[str], str],
) -> Callable[P, TestData]:
    def generate_data(L: int, N: int, sampling: str) -> dict[str, np.ndarray]:
        import so3

        so3_parameters = so3.create_parameter_dict(
            L=L,
            N=N,
            sampling_scheme_str=_s2fft_to_so3_sampling(sampling),
        )

        return {
            "f_size": so3.f_size(so3_parameters),
            "flmn_size": so3.flmn_size(so3_parameters),
            "n_alpha": so3.n_alpha(so3_parameters),
            "n_beta": so3.n_beta(so3_parameters),
            "n_gamma": so3.n_gamma(so3_parameters),
            "elmn2ind": {
                f"{el}_{m}_{n}": so3.elmn2ind(el, m, n, so3_parameters)
                for el in range(L)
                for m in range(-el, el + 1)
                for n in range(-N + 1, N)
            },
        }

    return cached_test_case_wrapper(generate_data, "json")


@pytest.fixture
def cached_ssht_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flm_generator: Callable[..., np.ndarray],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, L_lower: int, spin: int, sampling: str, reality: bool
    ) -> dict[str, np.ndarray]:
        import pyssht

        from s2fft.sampling import s2_samples

        flm = flm_generator(L=L, L_lower=L_lower, spin=spin, reality=reality)
        f_ssht = pyssht.inverse(
            s2_samples.flm_2d_to_1d(flm, L),
            L,
            Method=sampling.upper(),
            Spin=spin,
            Reality=reality if spin == 0 else False,
        )
        return {"flm": flm, "f_ssht": f_ssht}

    return cached_test_case_wrapper(generate_data, "npz")


@pytest.fixture
def cached_healpy_test_case(
    cached_test_case_wrapper: Callable[
        [Callable[P, TestData], str], Callable[P, TestData]
    ],
    flm_generator: Callable[..., np.ndarray],
) -> Callable[P, TestData]:
    def generate_data(
        L: int, nside: int, reality: bool, n_iter: int = 0
    ) -> dict[str, np.ndarray]:
        import healpy

        from s2fft.sampling import s2_samples

        flm = flm_generator(L=L, spin=0, reality=True)
        flm_hp = s2_samples.flm_2d_to_hp(flm, L)
        f_hp = healpy.sphtfunc.alm2map(flm_hp, nside, lmax=L - 1)
        flm_hp = healpy.sphtfunc.map2alm(f_hp, lmax=L - 1, iter=n_iter)
        return {"flm": flm, "f_hp": f_hp, "flm_hp": flm_hp}

    return cached_test_case_wrapper(generate_data, "npz")
