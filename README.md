[![Tests status](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml)
[![Linting status](https://github.com/astro-informatics/s2fft/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/linting.yml)
[![Documentation status](https://github.com/astro-informatics/s2fft/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/docs.yml)
[![Codecov](https://codecov.io/gh/astro-informatics/s2fft/branch/main/graph/badge.svg?token=7QYAFAAWLE)](https://codecov.io/gh/astro-informatics/s2fft)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI package](https://badge.fury.io/py/s2fft.svg)](https://badge.fury.io/py/s2fft)
[![arXiv](http://img.shields.io/badge/arXiv-2311.14670-orange.svg?style=flat)](https://arxiv.org/abs/2311.14670)<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-11-orange.svg?style=flat-square)](#contributors-)<!-- ALL-CONTRIBUTORS-BADGE:END --> 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astro-informatics/s2fft/blob/main/notebooks/spherical_harmonic_transform.ipynb)
[![Linter](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<img align="left" height="85" width="98" src="./docs/assets/sax_logo.png">

# Differentiable and accelerated spherical transforms

`S2FFT` is a Python package for computing Fourier transforms on the sphere
and rotation group [(Price & McEwen 2023)](https://arxiv.org/abs/2311.14670) using 
JAX or PyTorch. It leverages autodiff to provide differentiable transforms, which are 
also deployable on hardware accelerators (e.g. GPUs and TPUs).

More specifically, `S2FFT` provides support for spin spherical harmonic
and Wigner transforms (for both real and complex signals), with support
for adjoint transformations where needed, and comes with different
optimisations (precompute or not) that one may select depending on
available resources and desired angular resolution $L$.

> [!IMPORTANT]
> HEALPix long JIT compile time fixed for CPU!  Fix for GPU coming soon.

> [!TIP]
As of version 1.0.2 `S2FFT` also provides PyTorch implementations of underlying 
precompute transforms. In future releases this support will be extended to our 
on-the-fly algorithms.

> [!TIP]
As of version 1.1.0 `S2FFT` also provides JAX support for existing C/C++ packages, 
specifically `HEALPix` and `SSHT`. This works by wrapping python bindings with custom 
JAX frontends. Note that currently this C/C++ to JAX interoperability is currently 
limited to CPU.

## Algorithms :zap:

`S2FFT` leverages new algorithmic structures that can he highly
parallelised and distributed, and so map very well onto the architecture
of hardware accelerators (i.e. GPUs and TPUs). In particular, these
algorithms are based on new Wigner-d recursions that are stable to high
angular resolution $L$. The diagram below illustrates the recursions
(for further details see Price & McEwen, in prep.).

![image](./docs/assets/figures/Wigner_recursion_legend_darkmode.png)
With this recursion to hand, the spherical harmonic coefficients of an 
isolatitudinally sampled map may be computed as a two step process. First, 
a 1D Fourier transform over longitude, for each latitudinal ring. Second, 
a projection onto the real polar-d functions. One may precompute and store 
all real polar-d functions for extreme acceleration, however this comes 
with an equally extreme memory overhead, which is infeasible at L ~ 1024. 
Alternatively, the real polar-d functions may calculated recursively, 
computing only a portion of the projection at a time, hence incurring 
negligible memory overhead at the cost of slightly slower execution. The 
diagram below illustrates the separable spherical harmonic transform 
(for further details see Price & McEwen, in prep.).

![image](./docs/assets/figures/sax_schematic_legend_darkmode.png)

## Sampling :earth_africa:

The structure of the algorithms implemented in `S2FFT` can support any
isolatitude sampling scheme. A number of sampling schemes are currently
supported.

The equiangular sampling schemes of [McEwen & Wiaux
(2012)](https://arxiv.org/abs/1110.6298), [Driscoll & Healy
(1995)](https://www.sciencedirect.com/science/article/pii/S0196885884710086) 
and [Gauss-Legendre (1986)](https://link.springer.com/article/10.1007/BF02519350)
are supported, which exhibit associated sampling theorems and so
harmonic transforms can be computed to machine precision. Note that the
McEwen & Wiaux sampling theorem reduces the Nyquist rate on the sphere
by a factor of two compared to the Driscoll & Healy approach, halving
the number of spherical samples required.

The popular [HEALPix](https://healpix.jpl.nasa.gov) sampling scheme
([Gorski et al. 2005](https://arxiv.org/abs/astro-ph/0409513)) is also
supported. The HEALPix sampling does not exhibit a sampling theorem and
so the corresponding harmonic transforms do not achieve machine
precision but exhibit some error. However, the HEALPix sampling provides
pixels of equal areas, which has many practical advantages.

<p align="center"><img src="./docs/assets/figures/spherical_sampling.png" width="700"></p>

> [!NOTE]  
> For algorithmic reasons JIT compilation of HEALPix transforms can become slow at high bandlimits, due to XLA unfolding of loops which currently cannot be avoided. After compiling HEALPix transforms should execute with the efficiency outlined in the associated paper, therefore this additional time overhead need only be incurred once. We are aware of this issue and are working to fix it.  A fix for CPU execution has now been implemented (see example [notebook](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_HEALPix_backend.html)).  Fix for GPU execution is coming soon.

## Installation :computer:

The Python dependencies for the `S2FFT` package are listed in the file
`requirements/requirements-core.txt` and will be automatically installed
into the active python environment by [pip](https://pypi.org) when running

``` bash
pip install s2fft
```
This will install all core functionality which includes JAX support (including PyTorch support).

Alternatively, the `S2FFT` package may be installed directly from GitHub by cloning this 
repository and then running 

``` bash
pip install .        
```

from the root directory of the repository. 

Unit tests can then be executed to ensure the installation was successful by first installing the test requirements and then running pytest

``` bash
pip install -r requirements/requirements-tests.txt
pytest tests/  
```

Documentation for the released version is available [here](https://astro-informatics.github.io/s2fft/).  To build the documentation locally run

``` bash
pip install -r requirements/requirements-docs.txt
cd docs 
make html
open _build/html/index.html
```

> [!NOTE]  
> For plotting functionality which can be found throughout our various notebooks, one must install the requirements which can be found in `requirements/requirements-plotting.txt`.

## Usage :rocket:

To import and use `S2FFT` is as simple follows:

For a signal on the sphere

``` python
# Compute harmonic coefficients
flm = s2fft.forward_jax(f, L)  
# Map back to pixel-space signal
f = s2fft.inverse_jax(flm, L)
```

For a signal on the rotation group 

``` python
# Compute Wigner coefficients
flmn = s2fft.wigner.forward_jax(f, L, N)
# Map back to pixel-space signal
f = fft.wigner.inverse_jax(flmn, L, N)
```

For further details on usage see the [documentation](https://astro-informatics.github.io/s2fft/) and associated [notebooks](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/spherical_harmonic_transform.html).

> [!NOTE]  
> We also provide PyTorch support for the precompute version of our transforms. These are called through forward/inverse_torch(). Full PyTorch support will be provided in future releases.

## C/C++ JAX Frontends for SSHT/HEALPix :bulb:

`S2FFT` also provides JAX support for existing C/C++ packages, specifically [`HEALPix`](https://healpix.jpl.nasa.gov) and [`SSHT`](https://github.com/astro-informatics/ssht). This works 
by wrapping python bindings with custom JAX frontends. Note that this C/C++ to JAX interoperability is currently limited to CPU.

For example, one may call these alternate backends for the spherical harmonic transform by:

``` python
# Forward SSHT spherical harmonic transform
flm = s2fft.forward(f, L, sampling=["mw"], method="jax_ssht")  

# Forward HEALPix spherical harmonic transform
flm = s2fft.forward(f, L, nside=nside, sampling="healpix", method="jax_healpy")  
```

All of these JAX frontends supports out of the box reverse mode automatic differentiation, 
and under the hood is simply linking to the C/C++ packages you are familiar with. In this 
way `S2fft` enhances existing packages with gradient functionality for modern scientific computing or machine learning 
applications!

For further details on usage see the associated [notebooks](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_SSHT_backend.html).

<!-- ## Benchmarking :hourglass_flowing_sand:

We benchmarked the spherical harmonic and Wigner transforms implemented
in `S2FFT` against the C implementations in the
[SSHT](https://github.com/astro-informatics/ssht) package.

A brief summary is shown in the table below for the recursion (left) and
precompute (right) algorithms, with `S2FFT` running on GPUs (for further
details see Price & McEwen, in prep.). Note that our compute time is
agnostic to spin number (which is not the case for many other methods
that scale linearly with spin).

| L    | Wall-Time | Speed-up | Error    | Wall-Time | Speed-up | Error    | Memory  |
|------|-----------|----------|----------|-----------|----------|----------|---------|
| 64   | 3.6 ms    | 0.88     | 1.81E-15 | 52.4 μs   | 60.5     | 1.67E-15 | 4.2 MB  |
| 128  | 7.26 ms   | 1.80     | 3.32E-15 | 162 μs    | 80.5     | 3.64E-15 | 33 MB   |
| 256  | 17.3 ms   | 6.32     | 6.66E-15 | 669 μs    | 163      | 6.74E-15 | 268 MB  |
| 512  | 58.3 ms   | 11.4     | 1.43E-14 | 3.6 ms    | 184      | 1.37E-14 | 2.14 GB |
| 1024 | 194 ms    | 32.9     | 2.69E-14 | 32.6 ms   | 195      | 2.47E-14 | 17.1 GB |
| 2048 | 1.44 s    | 49.7     | 5.17E-14 | N/A       | N/A      | N/A      | N/A     |
| 4096 | 8.48 s    | 133.9    | 1.06E-13 | N/A       | N/A      | N/A      | N/A     |
| 8192 | 82 s      | 110.8    | 2.14E-13 | N/A       | N/A      | N/A      | N/A     |

where the left hand results are for the recursive based algorithm and the right hand side are 
our precompute implementation. -->

## Contributors ✨

Thanks goes to these wonderful people ([emoji
key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://cosmomatt.github.io"><img src="https://avatars.githubusercontent.com/u/32554533?v=4?s=100" width="100px;" alt="Matt Price"/><br /><sub><b>Matt Price</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=CosmoMatt" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3ACosmoMatt" title="Reviewed Pull Requests">👀</a> <a href="#ideas-CosmoMatt" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=jasonmcewen" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Ajasonmcewen" title="Reviewed Pull Requests">👀</a> <a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://matt-graham.github.io"><img src="https://avatars.githubusercontent.com/u/6746980?v=4?s=100" width="100px;" alt="Matt Graham"/><br /><sub><b>Matt Graham</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=matt-graham" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Amatt-graham" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sfmig.github.io/"><img src="https://avatars.githubusercontent.com/u/33267254?v=4?s=100" width="100px;" alt="sfmig"/><br /><sub><b>sfmig</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=sfmig" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Asfmig" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Devaraj-G"><img src="https://avatars.githubusercontent.com/u/36169767?v=4?s=100" width="100px;" alt="Devaraj Gopinathan"/><br /><sub><b>Devaraj Gopinathan</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=Devaraj-G" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://flanusse.net"><img src="https://avatars.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt="Francois Lanusse"/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=EiffL" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/issues?q=author%3AEiffL" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eltociear"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt="Ikko Eltociear Ashimine"/><br /><sub><b>Ikko Eltociear Ashimine</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=eltociear" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kmulderdas"><img src="https://avatars.githubusercontent.com/u/33317219?v=4?s=100" width="100px;" alt="Kevin Mulder"/><br /><sub><b>Kevin Mulder</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3Akmulderdas" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PhilippMisofCH"><img src="https://avatars.githubusercontent.com/u/142883157?v=4?s=100" width="100px;" alt="Philipp Misof"/><br /><sub><b>Philipp Misof</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3APhilippMisofCH" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ElisR"><img src="https://avatars.githubusercontent.com/u/19764906?v=4?s=100" width="100px;" alt="Elis Roberts"/><br /><sub><b>Elis Roberts</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3AElisR" title="Bug reports">🐛</a> <a href="https://github.com/astro-informatics/s2fft/commits?author=ElisR" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ASKabalan"><img src="https://avatars.githubusercontent.com/u/83787080?v=4?s=100" width="100px;" alt="Wassim KABALAN"/><br /><sub><b>Wassim KABALAN</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=ASKabalan" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3AASKabalan" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/astro-informatics/s2fft/commits?author=ASKabalan" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
We encourage contributions from any interested developers. A simple
first addition could be adding support for more spherical sampling
patterns!

## Attribution :books: 

Should this code be used in any way, we kindly request that the following article is
referenced. A BibTeX entry for this reference may look like:

``` 
@article{price:s2fft, 
   author      = "Matthew A. Price and Jason D. McEwen",
   title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
   journal     = "Journal of Computational Physics, submitted",
   year        = "2023",
   eprint      = "arXiv:2311.14670"        
}
```

You might also like to consider citing our related papers on which this
code builds:

``` 
@article{mcewen:fssht,
    author      = "Jason D. McEwen and Yves Wiaux",
    title       = "A novel sampling theorem on the sphere",
    journal     = "IEEE Trans. Sig. Proc.",
    year        = "2011",
    volume      = "59",
    number      = "12",
    pages       = "5876--5887",        
    eprint      = "arXiv:1110.6298",
    doi         = "10.1109/TSP.2011.2166394"
}
```

``` 
@article{mcewen:so3,
    author      = "Jason D. McEwen and Martin B{\"u}ttner and Boris ~Leistedt and Hiranya V. Peiris and Yves Wiaux",
    title       = "A novel sampling theorem on the rotation group",
    journal     = "IEEE Sig. Proc. Let.",
    year        = "2015",
    volume      = "22",
    number      = "12",
    pages       = "2425--2429",
    eprint      = "arXiv:1508.03101",
    doi         = "10.1109/LSP.2015.2490676"    
}
```

## License :memo:

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2023 Matthew Price, Jason McEwen and contributors.

`S2FFT` is free software made available under the MIT License. For
details see the [`LICENCE.txt`](LICENCE.txt) file.

The file [`lib/include/kernel_helpers.h`](lib/include/kernel_helpers.h) is adapted from
[code](https://github.com/dfm/extending-jax/blob/c33869665236877a2ae281f3f5dbff579e8f5b00/lib/kernel_helpers.h) in [a tutorial on extending JAX](https://github.com/dfm/extending-jax) by 
[Dan Foreman-Mackey](https://github.com/dfm) and licensed under a [MIT license](https://github.com/dfm/extending-jax/blob/371dca93c6405368fa8e71690afd3968d75f4bac/LICENSE).

The file [`lib/include/kernel_nanobind_helpers.h`](lib/include/kernel_nanobind_helpers.h)
is adapted from [code](https://github.com/jax-ml/jax/blob/3d389a7fb440c412d95a1f70ffb91d58408247d0/jaxlib/kernel_nanobind_helpers.h) 
by the [JAX](https://github.com/jax-ml/jax) authors 
and licensed under a [Apache-2.0 license](https://github.com/jax-ml/jax/blob/3d389a7fb440c412d95a1f70ffb91d58408247d0/LICENSE). 
