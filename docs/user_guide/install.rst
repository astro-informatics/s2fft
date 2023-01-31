.. _install:

Installation
============
There are two primary ways to install ``S2FFT``. One can either build the project from 
the most recent GitHub source, which comes with the added benefit of being able to 
locally execute the unit testing. Alternately, one may simply install the package directly 
from PyPi, an online python package manager.

Quick install (PyPi)
--------------------
Install ``S2FFT`` from PyPi with a single command

.. code-block:: bash

    pip install s2fft

Check that the package has installed by running pip list and locating ``S2FFT``.

Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n "env_name" python>=3.8
    conda activate "env_name"

Once within a fresh environment ``S2FFT`` may be installed by cloning the GitHub repository 
and pip installing locally

.. code-block:: bash

    git clone https://github.com/astro-informatics/s2fft
    cd s2fft
    pip install .
