cmake_minimum_required(VERSION 3.19...3.25)

# CUDA plugin of s2fft, built by the packages s2fft-cuda/cuda12 (s2fft-cuda12-plugin) and
# s2fft-cuda/cuda13 (s2fft-cuda13-plugin) with scikit-build-core. The architectures are taken from CMAKE_CUDA_ARCHITECTURES when it is
# given (the wheel builds pass the list explicitly); the default below is set before the
# CUDA language is enabled, so that it takes effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

project(s2fft_cuda LANGUAGES CXX CUDA)

message(STATUS "Using CMake version: ${CMAKE_VERSION}")

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT SKBUILD)
  message(
    FATAL_ERROR
      "Building standalone project directly without pip install is not supported. "
      "Please build the s2fft-cuda12-plugin or s2fft-cuda13-plugin package with pip.")
endif()

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER} (CUDA ${CUDAToolkit_VERSION})")

find_package(
  Python 3.11 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Detect the installed nanobind package and import it into CMake
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _s2fft
  STABLE_ABI
  ${CMAKE_CURRENT_LIST_DIR}/lib/src/extensions.cc
  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft.cu
  ${CMAKE_CURRENT_LIST_DIR}/lib/src/plan_cache.cc
  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_kernels.cu)

# cudart is linked statically; cuFFT dynamically, from the NVIDIA pip package of the
# with-cuda extra (found through the RPATH below) or from the system CUDA installation.
target_link_libraries(_s2fft PRIVATE CUDA::cudart_static CUDA::cufft)
target_include_directories(
  _s2fft PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include ${XLA_DIR} ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(_s2fft PROPERTIES LINKER_LANGUAGE CUDA
                                        CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(_s2fft PROPERTIES CUDA_ARCHITECTURES
                                        "${CMAKE_CUDA_ARCHITECTURES}")

# The plugin package: s2fft_cuda12_plugin or s2fft_cuda13_plugin, from the CUDA major
# version. cuFFT comes from nvidia-cufft-cu12 (nvidia/cufft/lib/libcufft.so.11) for CUDA
# 12 and from nvidia-cufft (nvidia/cu13/lib/libcufft.so.12) for CUDA 13; both install
# next to the plugin in site-packages.
set(S2FFT_CUDA_MAJOR ${CUDAToolkit_VERSION_MAJOR})
if(S2FFT_CUDA_MAJOR EQUAL 12)
  set(S2FFT_CUFFT_PIP "nvidia-cufft-cu12")
  set(S2FFT_CUFFT_SUBDIR "cufft/lib")
  set(S2FFT_CUFFT_SONAME "libcufft.so.11")
elseif(S2FFT_CUDA_MAJOR EQUAL 13)
  set(S2FFT_CUFFT_PIP "nvidia-cufft")
  set(S2FFT_CUFFT_SUBDIR "cu13/lib")
  set(S2FFT_CUFFT_SONAME "libcufft.so.12")
else()
  message(FATAL_ERROR "s2fft plugins are built with CUDA 12 or 13, found ${CUDAToolkit_VERSION}")
endif()
if(DEFINED S2FFT_PLUGIN_CUDA_MAJOR AND NOT S2FFT_PLUGIN_CUDA_MAJOR EQUAL S2FFT_CUDA_MAJOR)
  message(
    FATAL_ERROR
      "The package s2fft-cuda${S2FFT_PLUGIN_CUDA_MAJOR}-plugin is being built with CUDA "
      "${CUDAToolkit_VERSION}; use a CUDA ${S2FFT_PLUGIN_CUDA_MAJOR} toolkit.")
endif()
set(S2FFT_PLUGIN_PACKAGE "s2fft_cuda${S2FFT_CUDA_MAJOR}_plugin")
message(STATUS "Plugin package: ${S2FFT_PLUGIN_PACKAGE}, cuFFT ${S2FFT_CUFFT_SONAME} from ${S2FFT_CUFFT_PIP}")

set_target_properties(_s2fft PROPERTIES INSTALL_RPATH "$ORIGIN/../nvidia/${S2FFT_CUFFT_SUBDIR}")
configure_file(${CMAKE_CURRENT_LIST_DIR}/python/__init__.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/__init__.py @ONLY)
install(TARGETS _s2fft LIBRARY DESTINATION ${S2FFT_PLUGIN_PACKAGE})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION ${S2FFT_PLUGIN_PACKAGE})
