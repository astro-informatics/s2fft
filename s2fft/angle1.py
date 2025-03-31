import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)

import s2fft
from s2fft.base_transforms import spherical
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature, quadrature_jax, quadrature_torch


# "cc",mw, "mwss", "dh", "gl"


L =2;  spin=0; sampling = "gl"; reality ="true"
flm = s2fft.utils.signal_generator.generate_flm(np.random.default_rng(12345), L)
integral = flm[0, 0 + L - 1] * np.sqrt(4 * np.pi)
f = spherical.inverse(flm, L, spin, sampling, reality=reality)
q = quadrature.quad_weights(L, sampling, spin)
q = np.reshape(q, (-1, 1))


thetas_mw = s2fft.sampling.s2_samples.thetas(L, sampling=sampling)
nphi = samples.nphi_equiang(L, sampling)


Q = q.dot(np.ones((1, nphi)))
integral_check = np.sum(Q * f)

print()

print(integral_check)

