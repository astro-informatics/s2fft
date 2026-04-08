"""
Rotate a signal
===============

This tutorial demonstrates how to use ``S2FFT`` to rotate a signal on the sphere.

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
# A signal can be rotated in pixel space, however this can introduce artifacts.
# The best way to perform a rotation is through spherical harmonic space using the Wigner d-matrices, that is:
#
# * forward spherical harmonic transform,
# * rotation on the flm coefficients,
# * inverse spherical harmonic transform.
#
# Specifically, we will adopt the sampling scheme of `McEwen & Wiaux (2012) <https://arxiv.org/abs/1110.6298>`_.
# For our purposes here we'll just generate a random bandlimited signal.

import jax
import numpy as np

import s2fft

jax.config.update("jax_enable_x64", True)

L = 128
sampling = "mw"
rng = np.random.default_rng(12346161)
flm = s2fft.utils.signal_generator.generate_flm(rng, L)
f = s2fft.inverse(flm, L)

# %%
# Execute the rotation steps
# --------------------------
#
# First, we will run the ``JAX`` function to compute the spherical harmonic transform of our signal

flm = s2fft.forward_jax(f, L, reality=True)

# %%
# Now apply the rotation (here :math:`\pi/2` in each of ``alpha``, ``beta``, ``gamma``) on the harmonic coefficients ``flm``.

flm_rotated = s2fft.rotate_flms(flm, L, (np.pi / 2, np.pi / 2, np.pi / 2))

# %%
# Finally, we will run the ``JAX`` function to compute the inverse spherical harmonic transform to get back to pixel space.

f_rotated = s2fft.inverse_jax(flm_rotated, L, reality=True)
