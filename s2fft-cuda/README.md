# s2fft CUDA plugins

CUDA extension of [s2fft](https://github.com/astro-informatics/s2fft), which runs the HEALPix ring FFTs of `method="jax_cuda"` as JAX FFI custom calls.
The sources in this directory build two packages, `s2fft-cuda12-plugin` with CUDA 12 and `s2fft-cuda13-plugin` with CUDA 13, following `jax-cuda12-plugin` and `jax-cuda13-plugin`.

Install a plugin through the extras of s2fft, which pin it to the s2fft version:

```bash
pip install "s2fft[cuda12]"        # cuFFT from NVIDIA's pip packages
pip install "s2fft[cuda12-local]"  # cuFFT from the system CUDA
```

and the same with `cuda13`.
To build a plugin from a clone of the repository, with a CUDA 12 or 13 toolkit and CMake 3.19+, run

```bash
pip install -v ./s2fft-cuda/cuda12   # or ./s2fft-cuda/cuda13
```
