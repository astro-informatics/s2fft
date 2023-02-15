ls:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2FFT`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2FFT`` package is structured such that the 2 primary transforms, the Wigner and 
spherical harmonic transforms, can easily be accessed.

Usage |:rocket:|
-----------------
To import and use ``S2FFT``  is as simple follows: 

+-------------------------------------------------------+------------------------------------------------------------+
|For a signal on the sphere                             |For a signal on the rotation group                          |
|                                                       |                                                            |
|.. code-block:: Python                                 |.. code-block:: Python                                      |
|                                                       |                                                            |
|   # Compute harmonic coefficients                     |   # Compute Wigner coefficients                            |
|   flm = s2fft.forward_jax(f, L)                       |   flmn = s2fft.wigner.forward_jax(f, L, N)                 |
|                                                       |                                                            |
|   # Map back to pixel-space signal                    |   # Map back to pixel-space signal                         |
|   f = s2fft.inverse_jax(flm, L)                       |   f = s2fft.wigner.inverse_jax(flmn, L, N)                 |
+-------------------------------------------------------+------------------------------------------------------------+


Benchmarking |:hourglass_flowing_sand:|
-------------------------------------
We benchmarked the spherical harmonic and Wigner transforms implemented in ``S2FFT``
against the C implementations in the `SSHT <https://github.com/astro-informatics/ssht>`_
pacakge. 

A brief summary is shown in the table below for the recursion (left) and precompute
(right) algorithms, with ``S2FFT`` running on GPUs (for further details see Price &
McEwen, in prep.).  Note that our compute time is agnostic to spin number (which is not
the case for many other methods that scale linearly with spin).

+------+-----------+-----------+----------+-----------+----------+----------+---------+
|      |       Recursive Algorithm        |       Precompute Algorithm                |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| L    | Wall-Time | Speed-up  | Error    | Wall-Time | Speed-up | Error    | Memory  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 64   | 3.6 ms    | 0.88      | 1.81E-15 | 52.4 μs   | 60.5     | 1.67E-15 | 4.2 MB  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 128  | 7.26 ms   | 1.80      | 3.32E-15 | 162 μs    | 80.5     | 3.64E-15 | 33 MB   |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 256  | 17.3 ms   | 6.32      | 6.66E-15 | 669 μs    | 163      | 6.74E-15 | 268 MB  |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 512  | 58.3 ms   | 11.4      | 1.43E-14 | 3.6 ms    | 184      | 1.37E-14 | 2.14 GB |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 1024 | 194 ms    | 32.9      | 2.69E-14 | 32.6 ms   | 195      | 2.47E-14 | 17.1 GB |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 2048 | 1.44 s    | 49.7      | 5.17E-14 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 4096 | 8.48 s    | 133.9     | 1.06E-13 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+
| 8192 | 82 s      | 110.8     | 2.14E-13 | N/A       | N/A      | N/A      | N/A     |
+------+-----------+-----------+----------+-----------+----------+----------+---------+


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   spherical_harmonic/spherical_harmonic_transform.nblink
   wigner/wigner_transform.nblink
