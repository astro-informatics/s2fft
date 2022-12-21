Benchmarks for `s2fft`
=================================================================================================================

Benchmarking ``ss2ft`` with ``timeit``.


Description
-----------

The benchmark scripts are arranged as follows::

    .
    ├── README.rst
    ├── __init__.py
    ├── bench_wigner.py
    ├── bench_transforms.py    
    └── utils.py

| ``bench_wigner`` contains benchmarks for Wigner recursions
| ``bench_transforms`` contains benchmarks for forward transform

Usage
-----

| Inside the ``benchmarks`` directory run the appropriate ``bench_*`` script.
| *e.g.* to run benchmarks for recursions, run ``python bench_wigner.py``