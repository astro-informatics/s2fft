"""
Torch frontend guide
====================

This minimal tutorial demonstrates how to use the torch frontend for ``S2FFT`` to compute spherical harmonic transforms.

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
# Though ``S2FFT`` is primarily designed for ``JAX``, this torch functionality is fully unit tested (including gradients) and can be used straightforwardly as a learnable layer within existing models.
# As the torch functions wrap the ``JAX`` implementations we need to configure ``JAX`` to use 64-bit precision floating point types by default to ensure sufficient precision for the transforms - ``S2FFT`` will emit a warning if this has not been done.

import jax
import numpy as np
import torch

jax.config.update("jax_enable_x64", True)

from s2fft.transforms.spherical import forward, inverse
from s2fft.utils import signal_generator

# %%
# Lets set up a mock problem by specifying a bandlimit $L$ and generating some arbitrary harmonic coefficients.

L = 64
rng = np.random.default_rng(1234951510)
flm = torch.from_numpy(signal_generator.generate_flm(rng, L))

# %%
# Now lets calculate the signal on the sphere by applying the inverse spherical harmonic transform.

f = inverse(flm, L, method="torch")

# %%
# To calculate the corresponding spherical harmonic representation execute:

flm_check = forward(f, L, method="torch")

# %%
# Finally, lets check the error on the round trip is as expected for 64 bit machine precision floating point arithmetic.

print(f"Mean absolute error = {np.nanmean(np.abs(flm_check - flm))}")
