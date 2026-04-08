"""
JAX HEALPix Frontend
====================

This short tutorial demonstrates how to use the custom ``JAX`` frontend support ``S2FFT`` provides for the `HEALPix <https://healpix.jpl.nasa.gov>`_ C++ library.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :align: center
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/astro-informatics/s2fft/tree/gh-pages/_colab_notebooks/JAX_HEALPix_frontend.ipynb

If you are working on this notebook in Google Colab, you will need to have Google Colab install ``s2fft`` and ``healpy``.
You can do this by adding a cell to the top of the notebook with the following content:

.. code-block:: bash

    !pip install s2fft healpy &> /dev/null

and then running that cell.
"""

# %%
# ``S2FFT``'s support for the `HEALPix <https://healpix.jpl.nasa.gov>`_ C++ library resolves issues involving long JIT compile times for HEALPix when running on CPU.
# As with the other introductions, let's import some packages and define an arbitrary bandlimited signal to work with.
import jax
import numpy as np

import s2fft

jax.config.update("jax_enable_x64", True)

L = 128
nside = 64
method = "jax_healpy"
sampling = "healpix"
rng = np.random.default_rng(23457801234570)
flm = s2fft.utils.signal_generator.generate_flm(rng, L)
f = s2fft.inverse(flm, L, nside=nside, sampling=sampling, method=method)

# %%
# Calling forward HEALPix C++ function from JAX
# ---------------------------------------------

flm = s2fft.forward(f, L, nside=nside, sampling=sampling, method=method)

# %%
# Calling inverse HEALPix C++ function from JAX
# ---------------------------------------------

f_recov = s2fft.inverse(flm, L, nside=nside, sampling=sampling, method=method)

# %%
# Computing the roundtrip error
# -----------------------------
#
# Let's check the associated error, which should be around ``1e-5`` for healpix, which is not an exact sampling of the sphere.
# Note that increasing ``iters`` will reduce the numerical error here slightly, at the cost of linearly increased compute.

print(f"Mean absolute error = {np.nanmean(np.abs(f_recov - f))}")

# %%
# Differentiating through HEALPix C++ functions
# ---------------------------------------------
#
# So far all this is doing is providing an interface between ``JAX`` and ``HEALPix``, the real novelty comes when we differentiate through the C++ library.


# Define an arbitrary JAX function
def differentiable_test(flm) -> int:
    f = s2fft.inverse(flm, L, nside=nside, sampling=sampling, method=method)
    return jax.numpy.nanmean(jax.numpy.abs(f) ** 2)


# Create the JAX reverse mode gradient function
gradient_func = jax.grad(differentiable_test)

# Compute the gradient automatically
gradient = gradient_func(flm)

# %%
# Validating these gradients
# --------------------------
#
# This is all well and good, but how do we know these gradients are correct?
# Thankfully ``JAX`` provides a simple function to check this...

from jax.test_util import check_grads

check_grads(differentiable_test, (flm,), order=1, modes=("rev"))
