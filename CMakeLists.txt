cmake_minimum_required(VERSION 3.19...3.25)
project(s2fft LANGUAGES CXX)

message(STATUS "Using CMake version: ${CMAKE_VERSION}")

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

  if(NOT SKBUILD)
    message(FATAL_ERROR "Building standalone project directly without pip install is not supported"
                        "Please use pip install to build the project")
  else()
    find_package(CUDAToolkit REQUIRED)

    # Add the executable
    find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
    execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "-c" "from jax.extend import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
    message(STATUS "XLA include directory: ${XLA_DIR}")

    # Detect the installed nanobind package and import it into CMake
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_s2fft STABLE_ABI  
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/extensions.cc
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft.cu 
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_callbacks.cu
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/plan_cache.cc 
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft_kernels.cu 
                  )

    target_link_libraries(_s2fft PRIVATE CUDA::cudart_static CUDA::cufft_static CUDA::culibos)
    target_include_directories(_s2fft PUBLIC 
                                ${CMAKE_CURRENT_LIST_DIR}/lib/include 
                                ${XLA_DIR}
    ) 
    set_target_properties(_s2fft PROPERTIES 
                        LINKER_LANGUAGE CUDA
                        CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;89" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
    message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(_s2fft PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    install(TARGETS _s2fft LIBRARY DESTINATION s2fft_lib)
  endif()

else()
  if(SKBUILD)
    message(WARNING "CUDA compiler not found, building without CUDA support")

    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)  

    nanobind_add_module(_s2fft STABLE_ABI  
                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/extensions.cc
                  )

    target_compile_definitions(_s2fft PRIVATE NO_CUDA_COMPILER)
    target_include_directories(_s2fft PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include) 

    install(TARGETS _s2fft LIBRARY DESTINATION s2fft_lib)

  else()
    message(FATAL_ERROR "Building standalone project directly without pip install is not supported"
                        "Please use pip install to build the project")
  endif()
endif()


