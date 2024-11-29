# Benchmarks for `s2fft`

Scripts for benchmarking `s2fft` with `timeit` (and optionally `memory_profiler`).

Measures time to compute transforms for grids of parameters settings, optionally 
outputting the results to a JSON file to allow comparing performance over versions
and/or systems. 
If the [`memory_profiler` package](https://github.com/pythonprofilers/memory_profiler) 
is installed an estimate of the peak (main) memory usage of the benchmarked functions
will also be recorded.
If the [`py-cpuinfo` package](https://pypi.org/project/py-cpuinfo/) 
is installed additional information about CPU of system benchmarks are run on will be
recorded in JSON output.


## Description

The benchmark scripts are as follows:

  * `spherical.py` contains benchmarks for on-the-fly implementations of spherical 
    transforms (forward and inverse).
  * `precompute_spherical.py` contains benchmarks for precompute implementations of
    spherical transforms (forward and inverse).
  * `wigner.py` contains benchmarks for on-the-fly implementations of Wigner-d
    transforms (forward and inverse).
  * `precompute_wigner.py` contains benchmarks for precompute implementations of
    Wigner-d transforms (forward and inverse).
  
The `benchmarking.py` module contains shared utility functions for defining and running
the benchmarks.

## Usage

Each benchmark script defines a set of default parameter values to run the benchmarks
over. A set of command line arguments can be used to control the benchmark runs,
and optionally override parameter values benchmarked and specify a file to output
the JSON formatted benchmark results to. Pass a `--help` argument to the script to
display the usage message:

```
usage: spherical.py [-h] [-number-runs NUMBER_RUNS] [-repeats REPEATS]
                    [-parameter-overrides [PARAMETER_OVERRIDES ...]] [-output-file OUTPUT_FILE]
                    [--run-once-and-discard]

Benchmarks for on-the-fly spherical transforms.

options:
  -h, --help            show this help message and exit
  -number-runs NUMBER_RUNS
                        Number of times to run the benchmark in succession in each timing run. (default: 10)
  -repeats REPEATS      Number of times to repeat the benchmark runs. (default: 3)
  -parameter-overrides [PARAMETER_OVERRIDES ...]
                        Override for values to use for benchmark parameter. A parameter name followed by space
                        separated list of values to use. May be specified multiple times to override multiple
                        parameters. (default: None)
  -output-file OUTPUT_FILE
                        File path to write JSON formatted results to. (default: None)
  --run-once-and-discard
                        Run benchmark function once first without recording time to ignore the effect of any initial
                        one-off costs such as just-in-time compilation. (default: False)
```

For example to run the spherical transform benchmarks using only the JAX implementations,
running on a CPU (in double-precision) for `L` values 64, 128, 256, 512 and 1024 we 
would run from the root of the repository:

```sh
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python benchmarks/spherical.py --run-once-and-discard -p L 64 128 256 512 1024 -p method jax
```

Note the usage of environment variables `JAX_PLATFORM_NAME` and `JAX_ENABLE_X64` to 
configure the default device used by JAX and whether to enable double-precision
computations by default respectively.
