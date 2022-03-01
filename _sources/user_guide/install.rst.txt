.. _install:

Installation
============
Link to `PyPi <https://pypi.org>`_ and provide link for source install.

Quick install (PyPi)
--------------------
Install **<Project-name>** from PyPi with a single command

.. code-block:: bash

    pip install <Project-name>

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate <Project-name>.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n <Project-name>_env python=3.8
    conda activate <Project-name>_env

Once within a fresh environment **<Project-name>** may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/<Project-name>
    cd <Project-name>

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_<Project-name>.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest --black <Project-name>/tests/ 

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
