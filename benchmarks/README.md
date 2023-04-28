# Benchmarks for `s2fft`

Scripts for benchmarking `ss2ft` with `timeit` (and optionally `memory_profiler`).

## Description

The benchmark scripts are as follows:

  * `wigner.py` contains benchmarks for Wigner transforms (forward and inverse)
  * `spherical.py` contains benchmarks for spherical transforms (forward and inverse)

The `benchmarking.py` module contains shared utility functions for defining and running
the benchmarks.

## Usage

Each benchmark script defines a set of default parameter values to run the benchmarks
over. A set of command line arguments can be used to control the benchmark runs,
and optionally override parameter values benchmarked and specify a file to output
the JSON formatted benchmark results to. Pass a `--help` argument to the script to
display the usage message:

```
usage: Run benchmarks [-h] [-number-runs NUMBER_RUNS] [-repeats REPEATS]
                      [-parameter-overrides [PARAMETER_OVERRIDES [PARAMETER_OVERRIDES ...]]]
                      [-output-file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -number-runs NUMBER_RUNS
                        Number of times to run the benchmark in succession in each
                        timing run.
  -repeats REPEATS      Number of times to repeat the benchmark runs.
  -parameter-overrides [PARAMETER_OVERRIDES [PARAMETER_OVERRIDES ...]]
                        Override for values to use for benchmark parameter. A parameter
                        name followed by space separated list of values to use. May be
                        specified multiple times to override multiple parameters.
  -output-file OUTPUT_FILE
                        File path to write JSON formatted results to.
```

For example to run the spherical transform benchmarks using only the JAX implementations,
running on a CPU (in double-precision) for `L` values 64, 128, 256, 512 and 1024 we 
would run from the root of the repository:

```sh
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python benchmarks/spherical.py -p L 64 128 256 512 1024 -p method jax
```

Note the usage of environment variables `JAX_PLATFORM_NAME` and `JAX_ENABLE_X64` to 
configure the default device used by JAX and whether to enable double-precision
computations by default respectively.
