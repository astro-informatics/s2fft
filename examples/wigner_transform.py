"""
Wigner transform
================

This tutorial demonstrates how to use ``S2FFT`` to compute Wigner transforms, i.e. Fourier transforms on the rotation group :math:`SO(3)`.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :align: center
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/astro-informatics/s2fft/tree/gh-pages/_colab_notebooks/spherical_rotation.ipynb

If you are working on this notebook in Google Colab, you will need to have Google Colab install ``s2fft``.
You can do this by adding a cell to the top of the notebook with the following content:

.. code-block:: bash

    !pip install s2fft &> /dev/null

and then running that cell.
"""

# %%
# Specifically, we will adopt the sampling scheme of `McEwen et al. (2015) <https://arxiv.org/abs/1508.03101>`_.
#
# To demonstrate how to compute ``S2FFT`` Wigner transforms we will first construct an input signal that is sampled on the rotation group using this sampling scheme.
# We'll simply construct a random test signal in harmonic space for demonstration purposes.

import jax
import numpy as np

import s2fft

jax.config.update("jax_enable_x64", True)

L = 128
N = 3
reality = True
rng = np.random.default_rng(83459)
flmn = s2fft.utils.signal_generator.generate_flmn(rng, L, N, reality=reality)

# %%
# Computing the inverse Wigner transform
# --------------------------------------
#
# Let's run the ``JAX`` function to compute the inverse Wigner transform of this random signal.

f = s2fft.wigner.inverse_jax(flmn, L, N, reality=reality)

# %%
# If you are planning on applying this transform many times (e.g. during training of a model) we recommend precomputing and storing some small arrays that are used every time.
# To do this simply compute these and pass as a static argument.

precomps = s2fft.generate_precomputes_wigner_jax(L, N, forward=False, reality=reality)
f_pre = s2fft.wigner.inverse_jax(flmn, L, N, reality=reality, precomps=precomps)

# %%
# Computing the forward Wigner transform
# --------------------------------------
#
# Let's run the ``JAX`` function to compute the forward Wigner transforms to get us back to the random Wigner coefficients.

flmn_recov = s2fft.wigner.forward_jax(f, L, N, reality=reality)

# %%
# Again, if you are planning on applying this transform many times (e.g. during training of a model) we recommend precomputing and storing some small arrays that are used every time.
# To do this simply compute these and pass as a static argument.

precomps = s2fft.generate_precomputes_wigner_jax(L, N, forward=True, reality=reality)
flmn_recov_pre = s2fft.wigner.forward_jax(
    f_pre, L, N, reality=reality, precomps=precomps
)

# %%
# Computing the error
# -------------------
#
# Let's check the roundtrip error, which should be close to machine precision for the sampling theorem used.

print(f"Mean absolute error = {np.nanmean(np.abs(flmn_recov - flmn))}")
print(
    f"Mean absolute error using precomputes = {np.nanmean(np.abs(flmn_recov_pre - flmn))}"
)
