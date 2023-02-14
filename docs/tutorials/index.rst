:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2FFT`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2FFT`` package is structured such that the 2 primary transforms, the Wigner and 
spherical harmonic transforms, can easily be accessed in both a precomputed and recursive 
mode with memory overhead :math:`\mathcal{O}(L^3)` and :math:`\mathcal{O}(L^2)` respectively.

To import the core transforms one need only run 

.. code-block:: python 

   from s2fft.transforms import spherical, wigner 

or for the precompute transforms run 

.. code-block:: python 

   from s2fft.precompute_transforms import spherical, wigner 

To know which arguments must be passed to these functions check out the API section of the 
documentation; note that the function arguments are kept as simple as possible, so in 
many cases it will be as simple as, e.g.

.. code-block:: python 

   flm = spherical.forward(f, L, spin)
   f = spherical.inverse(flm, L, spin)

Another factor you will need to consider is ensuring that the shapes of input/output 
arrays are of the correct shape. Again there are functions to handle these requests 
provided in the ``S2FFT`` API documentation.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   spherical_harmonic/spherical_harmonic_transform.nblink
   wigner/wigner_transform.nblink
