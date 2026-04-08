"""
JAX SSHT frontend
=================

This short tutorial demonstrates how to use the custom ``JAX`` frontend support ``S2FFT`` provides for the `SSHT <https://github.com/astro-informatics/ssht>`_ C library.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :align: center
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/astro-informatics/s2fft/tree/gh-pages/_colab_notebooks/JAX_SSHT_frontend.ipynb

If you are working on this notebook in Google Colab, you will need to have Google Coab install ``s2fft`` and ``pyssht``.
You can do this by adding a cell to the top of the notebook with the following content:

.. code-block:: bash

    !pip install s2fft pyssht &> /dev/null

and then running that cell.
"""

# %%
# As with the other introductions, let's import some packages and define an arbitrary bandlimited signal to work with.

import jax
import numpy as np

import s2fft

jax.config.update("jax_enable_x64", True)

L = 128
method = "jax_ssht"
rng = np.random.default_rng(23457801234570)
flm = s2fft.utils.signal_generator.generate_flm(rng, L)
f = s2fft.inverse(flm, L, method=method)

# %%
# Calling forward SSHT C function from JAX
# ----------------------------------------

flm = s2fft.forward(f, L, method=method)

# %%
# Calling inverse SSHT C function from JAX
# ----------------------------------------

f_recov = s2fft.inverse(flm, L, method=method)

# %%
# Computing the roundtrip error
# -----------------------------
#
# Let's check the associated error, which should be close to machine precision for the sampling scheme used.

print(f"Mean absolute error = {np.nanmean(np.abs(f_recov - f))}")

# %%
# Differentiating through SSHT C functions
# ----------------------------------------
#
# So far all this is doing is providing an interface between ``JAX`` and ``SSHT``, the real novelty comes when we differentiate through the C library.


# Define an arbitrary JAX function
def differentiable_test(flm) -> int:
    f = s2fft.inverse(flm, L, method=method)
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
