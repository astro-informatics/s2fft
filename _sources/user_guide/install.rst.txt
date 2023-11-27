:html_theme.sidebar_secondary.remove:

.. _install:

Installation
=========================
There are two primary ways to install ``S2FFT``. One can either build the project from 
the most recent GitHub source, which comes with the added benefit of being able to 
locally execute the unit testing. Alternately, one may simply install the package directly 
from PyPi, an online python package manager.

Quick install (PyPi)
--------------------
The simplest way to pick up ``S2FFT`` is to install it directly from PyPi by running 

.. code-block:: bash
    
    pip install s2fft 

after which ``S2FFT`` may be imported and run as outlined in the associated notebooks and collab tutorials.

Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n "env_name" python>=3.9
    conda activate "env_name"

Once within a fresh environment ``S2FFT`` may be installed by cloning the GitHub repository 
and pip installing locally

.. code-block:: bash

    git clone https://github.com/astro-informatics/s2fft
    cd s2fft
    pip install .

from the root directory of the repository. Unit tests can then be executed to ensure the 
installation was successful by running 

.. code-block:: bash 

    pytest tests/ 

In the very near future one will be able to install ``S2FFT`` directly from `PyPi` by ``pip install s2fft`` but this is not yet supported.

Installing JAX for NVIDIA GPUs
------------------------------
We include both ``jax`` and ``jaxlib`` as dependencies in ``requirements/requirements-core.txt`` 
however to get things running on GPUs can be a bit more involved. We strongly recommend 
this installation `guide <https://github.com/google/jax#installation>`_ provided by 
Google. To summarise you will first need to install NVIDIA drivers for 
`CUDA <https://developer.nvidia.com/cuda-downloads>`_ and `CuDNN <https://developer.nvidia.com/CUDNN>`_, 
following which a pre-built CUDA-compatible wheels shoulld be installed by running 

.. code-block:: bash 

    pip install --upgrade pip 

    # Wheels only built for linux
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # Wheels built for many machine architectures 
    pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

where the versions of CUDA and CuDNN should match those you have installed on the machine.
