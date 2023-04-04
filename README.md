[![image](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2fft/actions/workflows/tests.yml)
[![image](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://astro-informatics.github.io/s2fft)
[![image](https://codecov.io/gh/astro-informatics/s2fft/branch/main/graph/badge.svg?token=7QYAFAAWLE)](https://codecov.io/gh/astro-informatics/s2fft)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat)](https://arxiv.org/abs/xxxx.xxxxx)<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-) <!-- ALL-CONTRIBUTORS-BADGE:END --> 
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img align="left" height="85" width="98" src="./docs/assets/sax_logo.png">

# Differentiable and accelerated spherical transforms with JAX

`S2FFT` is a JAX package for computing Fourier transforms on the sphere
and rotation group. It leverages autodiff to provide differentiable
transforms, which are also deployable on modern hardware accelerators
(e.g. GPUs and TPUs).

More specifically, `S2FFT` provides support for spin spherical harmonic
and Wigner transforms (for both real and complex signals), with support
for adjoint transformations where needed, and comes with different
optimisations (precompute or not) that one may select depending on
available resources and desired angular resolution $L$.

## Algorithms :zap:

`S2FFT` leverages new algorithmic structures that can he highly
parallelised and distributed, and so map very well onto the architecture
of hardware accelerators (i.e. GPUs and TPUs). In particular, these
algorithms are based on new Wigner-d recursions that are stable to high
angular resolution $L$. The diagram below illustrates the recursions
(for further details see Price & McEwen, in prep.).

![image](./docs/assets/figures/schematic.png)

## Sampling :earth_africa:

The structure of the algorithms implemented in `S2FFT` can support any
isolattitude sampling scheme. A number of sampling schemes are currently
supported.

The equiangular sampling schemes of [McEwen & Wiaux
(2012)](https://arxiv.org/abs/1110.6298) and [Driscoll & Healy
(1995)](https://www.sciencedirect.com/science/article/pii/S0196885884710086)
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

<p align="center"><img src="./docs/assets/figures/spherical_sampling.png" width="500"></p>

## Installation :computer:

The Python dependencies for the `S2FFT` package are listed in the file
`requirements/requirements-core.txt` and will be automatically installed
into the active python environment by [pip](https://pypi.org) when running

``` bash
pip install .        
```

from the root directory of the repository. Unit tests can then be
executed to ensure the installation was successful by running

``` bash
pytest tests/         # for pytest
tox -e py38           # for tox 
```

In the very near future one will be able to install `S2FFT` directly
from [PyPi](https://pypi.org) by `pip install s2fft` but this is not yet
supported. Note that to run `JAX` on NVIDIA GPUs you will need to follow
the [guide](https://github.com/google/jax#installation) outlined by
Google. 

    Note: For plotting functionality which can be found throughout our various notebooks, one 
    must install the requirements which can be found in `requirements/requirements-plotting.txt`.

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

## Benchmarking :hourglass_flowing_sand:

We benchmarked the spherical harmonic and Wigner transforms implemented
in `S2FFT` against the C implementations in the
[SSHT](https://github.com/astro-informatics/ssht) pacakge.

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
our precompute implementation.

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
   AUTHOR      = "Matthew A. Price and Jason D. McEwen and Contributors",
   TITLE       = "TBA",
   YEAR        = "2023",
   EPRINT      = "arXiv:0000.00000"        
}
```

You might also like to consider citing our related papers on which this
code builds:

``` 
@article{mcewen:fssht,
    AUTHOR      = "Jason D. McEwen and Yves Wiaux",
    TITLE       = "A novel sampling theorem on the sphere",
    JOURNAL     = "IEEE Trans. Sig. Proc.",
    YEAR        = "2011",
    VOLUME      = "59",
    NUMBER      = "12",
    PAGES       = "5876--5887",        
    EPRINT      = "arXiv:1110.6298",
    DOI         = "10.1109/TSP.2011.2166394"
}
```

``` 
@article{mcewen:so3,
    AUTHOR      = "Jason D. McEwen and Martin B{\"u}ttner and Boris ~Leistedt and Hiranya V. Peiris and Yves Wiaux",
    TITLE       = "A novel sampling theorem on the rotation group",
    JOURNAL     = "IEEE Sig. Proc. Let.",
    YEAR        = "2015",
    VOLUME      = "22",
    NUMBER      = "12",
    PAGES       = "2425--2429",
    EPRINT      = "arXiv:1508.03101",
    DOI         = "10.1109/LSP.2015.2490676"    
}
```

## License :memo:

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2023 Matthew Price, Jason McEwen and contributors.

`S2FFT` is free software made available under the MIT License. For
details see the LICENSE file.
