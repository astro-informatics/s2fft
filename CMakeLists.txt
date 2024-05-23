cmake_minimum_required(VERSION 3.19...3.25)

# NVCC 12 does not support C++20
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
# 70 for V100, 80 for A100, 89 for my laptop RTX 4060
set(CMAKE_CUDA_ARCHITECTURES "70;80;89" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
project(s2fft LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)


if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12 AND CUDAToolkit_VERSION_MINOR GREATER_EQUAL 2)
  message(STATUS "CUDA version is compatible (12.2 and above)")
else()
  message(STATUS "CUDA version is not compatible (requires 12.2 and above)")
endif()



if (NOT SKBUILD)

  message(WARNING "This is being compiled as a standalone project (without pip install)"
                  "\nTo compile as a python package, use the command: \n\tpython -m pip install s2fft"
                  "\nThis will create the testing executable only and is meant only for s2fft developpers")


  add_executable(_s2fftExe   
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft.cu 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/plan_cache.cc 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_kernels.cu 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_callbacks.cu
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/fft_test.cu
                )

  target_link_libraries(_s2fftExe PRIVATE CUDA::cudart_static CUDA::cufft_static CUDA::culibos)
  target_include_directories(_s2fftExe PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include) 
  set_target_properties(_s2fftExe PROPERTIES 
                      LINKER_LANGUAGE CUDA
                      CUDA_SEPARABLE_COMPILATION ON)
  message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set_target_properties(_s2fftExe PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  

else()

  # Print in green
  string(ASCII 27 Esc)
  message(STATUS "${Esc}[32mCompiling and installing s2fft primitives${Esc}[m")

  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)


  if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()

  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)


  nanobind_add_module(_s2fft STABLE_ABI  
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/extensions.cc
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft.cu 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_callbacks.cu
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/plan_cache.cc 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_kernels.cu 
                )


  target_link_libraries(_s2fft PRIVATE CUDA::cudart_static CUDA::cufft_static CUDA::culibos)
  target_include_directories(_s2fft PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include) 
  set_target_properties(_s2fft PROPERTIES 
                      LINKER_LANGUAGE CUDA
                      CUDA_SEPARABLE_COMPILATION ON)
  message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set_target_properties(_s2fft PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")


  install(TARGETS _s2fft LIBRARY DESTINATION s2fft_lib)


endif()
