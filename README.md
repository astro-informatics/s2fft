<div style="text-align: center;" align="center">

<img class="dark-light" width="98" height="85" alt="s2fft logo - schematic representation of a tiled sphere" src="https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/sax_logo.png">

# Differentiable and accelerated spherical transforms

[![Tests status](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml)
[![Linting status](https://github.com/astro-informatics/s2fft/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/linting.yml)
[![Documentation status](https://github.com/astro-informatics/s2fft/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/docs.yml)
[![Codecov](https://codecov.io/gh/astro-informatics/s2fft/branch/main/graph/badge.svg?token=7QYAFAAWLE)](https://codecov.io/gh/astro-informatics/s2fft)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI package](https://badge.fury.io/py/s2fft.svg)](https://badge.fury.io/py/s2fft)
[![arXiv](http://img.shields.io/badge/arXiv-2311.14670-orange.svg?style=flat)](https://arxiv.org/abs/2311.14670)
![All Contributors](https://img.shields.io/github/all-contributors/astro-informatics/s2fft?color=ee8449&style=flat-square)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astro-informatics/s2fft/blob/main/notebooks/spherical_harmonic_transform.ipynb)
[![Linter](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

`S2FFT` is a Python package for computing Fourier transforms on the sphere
and rotation group [(Price & McEwen 2024)](https://arxiv.org/abs/2311.14670) using 
JAX or PyTorch. It leverages autodiff to provide differentiable transforms, which are 
also deployable on hardware accelerators (e.g. GPUs and TPUs).

More specifically, `S2FFT` provides support for spin spherical harmonic
and Wigner transforms (for both real and complex signals), with support
for adjoint transformations where needed, and comes with different
optimisations (precompute or not) that one may select depending on
available resources and desired angular resolution $L$.

## Algorithms ⚡

`S2FFT` leverages new algorithmic structures that can he highly
parallelised and distributed, and so map very well onto the architecture
of hardware accelerators (i.e. GPUs and TPUs). In particular, these
algorithms are based on new Wigner-d recursions that are stable to high
angular resolution $L$. The diagram below illustrates the recursions
(for further details see [Price & McEwen 2024]((https://arxiv.org/abs/2311.14670))).

<div style="text-align: center;" align="center">
<img class="dark-light" alt="Schematic of Wigner recursions" src="https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/Wigner_recursion_legend_darkmode.png" />
</div>

With this recursion to hand, the spherical harmonic coefficients of an 
isolatitudinally sampled map may be computed as a two step process. First, 
a 1D Fourier transform over longitude, for each latitudinal ring. Second, 
a projection onto the real polar-d functions. One may precompute and store 
all real polar-d functions for extreme acceleration, however this comes 
with an equally extreme memory overhead, which is infeasible at $L \sim 1024$. 
Alternatively, the real polar-d functions may calculated recursively, 
computing only a portion of the projection at a time, hence incurring 
negligible memory overhead at the cost of slightly slower execution. The 
diagram below illustrates the separable spherical harmonic transform 
(for further details see [Price & McEwen 2024]((https://arxiv.org/abs/2311.14670))).

<div style="text-align: center;" align="center">
<img class="dark-light" alt="Schematic of forward and inverse spherical harmonic transforms" src="https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/sax_schematic_legend_darkmode.png" />
</div>

## Sampling 🌍

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

<div style="text-align: center;" align="center">
<img class="dark-light" alt="Visualization of spherical sampling schemes" src="https://raw.githubusercontent.com/astro-informatics/s2fft/main/docs/assets/figures/spherical_sampling.png" width="700">
</div>

> [!NOTE]  
> For algorithmic reasons JIT compilation of HEALPix transforms can become slow at high bandlimits, due to XLA unfolding of loops which currently cannot be avoided. After compiling HEALPix transforms should execute with the efficiency outlined in the associated paper, therefore this additional time overhead need only be incurred once. We are aware of this issue and are working to fix it.  A fix for CPU execution has now been implemented (see example [notebook](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_HEALPix_backend.html)).

## Installation 💻

The latest release of `S2FFT` published [on PyPI](https://pypi.org/project/s2fft/) can be installed by running

```bash
pip install s2fft
```

This will install `S2FFT`'s dependencies including JAX if not already installed.
As by default installing JAX from PyPI will use a CPU-only build,
if you wish to install JAX with GPU or TPU support,
you should first follow the [relevant installation instructions in JAX's documentation](https://docs.jax.dev/en/latest/installation.html#installation)
and then install `S2FFT` as above.

Alternatively, the  latest development version of `S2FFT` may be installed directly from GitHub by running 

```bash
pip install git+https://github.com/astro-informatics/s2fft  
```

## Tests 🚦

A `pytest` test suite for the package is included in the `tests` directory.
To install the test dependencies, clone the repository and install the package (in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)) 
with the extra test dependencies by running from the root of the repository

```bash
pip install -e ".[tests]"
```

To run the tests, run from the root of the repository

```bash
pytest  
```

## Documentation 📖

Documentation for the released version is available [here](https://astro-informatics.github.io/s2fft/). 
To install the documentation dependencies, clone the repository and install the package (in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)) 
with the extra documentation dependencies by running from the root of the repository

```bash
pip install -e ".[docs]"
```

To build the documentation, run from the root of the repository

```bash
cd docs 
make html
open _build/html/index.html
```

## Notebooks 📓

A series of tutorial notebooks are included in the `notebooks` directory
and rendered [in the documentation](https://astro-informatics.github.io/s2fft/tutorials/index.html).

To install the dependencies required to run the notebooks locally, clone the repository and install the package (in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)) 
with the extra documentation and plotting dependencies by running from the root of the repository

```bash
pip install -e ".[docs,plotting]"
```

To run the notebooks in Jupyter Lab, run from the root of the repository

```bash
jupyter lab
```

## Usage 🚀

To import and use `S2FFT` is as simple follows:

For a signal on the sphere

```python
import s2fft

# Define sampled signal to transform and harmonic bandlimit
f = ...
L = ...
# Compute harmonic coefficients
flm = s2fft.forward(f, L, method="jax")  
# Map back to pixel-space signal
f = s2fft.inverse(flm, L, method="jax")
```

For a signal on the rotation group 

```python
import s2fft

# Define sampled signal to transform and harmonic and azimuthal bandlimits
f = ...
L = ...
N = ...
# Compute Wigner coefficients
flmn = s2fft.wigner.forward(f, L, N, method="jax")
# Map back to pixel-space signal
f = fft.wigner.inverse_jax(flmn, L, N, method="jax")
```

For further details on usage see the [documentation](https://astro-informatics.github.io/s2fft/) and associated [notebooks](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/spherical_harmonic_transform.html).

> [!NOTE]  
> We also provide PyTorch support for the precompute version of our transforms, as demonstrated in the [_Torch frontend_ tutorial notebook](https://astro-informatics.github.io/s2fft/tutorials/torch_frontend/torch_frontend.html).

## SSHT & HEALPix wrappers 💡

`S2FFT` also provides JAX support for existing C/C++ packages, specifically [`HEALPix`](https://healpix.jpl.nasa.gov) and [`SSHT`](https://github.com/astro-informatics/ssht). This works 
by wrapping Python bindings with custom JAX frontends. Note that this C/C++ to JAX interoperability is currently limited to CPU.

For example, one may call these alternate backends for the spherical harmonic transform by:

``` python
# Forward SSHT spherical harmonic transform
flm = s2fft.forward(f, L, sampling="mw", method="jax_ssht")  

# Forward HEALPix spherical harmonic transform
flm = s2fft.forward(f, L, nside=nside, sampling="healpix", method="jax_healpy")  
```

All of these JAX frontends supports out of the box reverse mode automatic differentiation, 
and under the hood is simply linking to the C/C++ packages you are familiar with. In this 
way `S2fft` enhances existing packages with gradient functionality for modern scientific computing or machine learning 
applications!

For further details on usage see the associated [notebooks](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_SSHT_backend.html).

## Benchmarks ⏱️

A suite of benchmark functions for both the on-the-fly and precompute versions of the spherical harmonic and Wigner transforms are available in the `benchmarks` directory, along with utilities for running the benchmarks and plotting the results.

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PhilippMisofCH"><img src="https://avatars.githubusercontent.com/u/142883157?v=4?s=100" width="100px;" alt="Philipp Misof"/><br /><sub><b>Philipp Misof</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3APhilippMisofCH" title="Bug reports">🐛</a> <a href="https://github.com/astro-informatics/s2fft/commits?author=PhilippMisofCH" title="Documentation">📖</a></td>
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

## Attribution 📚

Should this code be used in any way, we kindly request that the following article is
referenced. A BibTeX entry for this reference may look like:

``` 
@article{price:s2fft, 
   author      = "Matthew A. Price and Jason D. McEwen",
   title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
   journal     = "Journal of Computational Physics",
   year        = "2024",
   volume      = "510",
   pages       = "113109",
   eprint      = "arXiv:2311.14670",
   doi         = "10.1016/j.jcp.2024.113109"
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

## License 📝

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2023 Matthew Price, Jason McEwen and contributors.

`S2FFT` is free software made available under the MIT License. For
details see the [`LICENCE.txt`](https://github.com/astro-informatics/s2fft/blob/main/LICENCE.txt) file.

The file [`lib/include/kernel_helpers.h`](https://github.com/astro-informatics/s2fft/blob/main/lib/include/kernel_helpers.h) is adapted from
[code](https://github.com/dfm/extending-jax/blob/c33869665236877a2ae281f3f5dbff579e8f5b00/lib/kernel_helpers.h) in [a tutorial on extending JAX](https://github.com/dfm/extending-jax) by 
[Dan Foreman-Mackey](https://github.com/dfm) and licensed under a [MIT license](https://github.com/dfm/extending-jax/blob/371dca93c6405368fa8e71690afd3968d75f4bac/LICENSE).

The file [`lib/include/kernel_nanobind_helpers.h`](https://github.com/astro-informatics/s2fft/blob/main/lib/include/kernel_nanobind_helpers.h)
is adapted from [code](https://github.com/jax-ml/jax/blob/3d389a7fb440c412d95a1f70ffb91d58408247d0/jaxlib/kernel_nanobind_helpers.h) 
by the [JAX](https://github.com/jax-ml/jax) authors 
and licensed under a [Apache-2.0 license](https://github.com/jax-ml/jax/blob/3d389a7fb440c412d95a1f70ffb91d58408247d0/LICENSE). 
