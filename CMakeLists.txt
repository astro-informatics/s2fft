cmake_minimum_required(VERSION 3.19...3.25)

# NVCC 12 does not support C++20
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
# 70 for V100, 80 for A100, 89 for my laptop RTX 4060
set(CMAKE_CUDA_ARCHITECTURES "70;80;89" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
project(s2fft LANGUAGES CXX CUDA)


#if (NOT SKBUILD)
#  message(WARNING "\
#  This CMake file is meant to be executed using 'scikit-build'. Running
#  it directly will almost certainly not produce the desired result. If
#  you are a user trying to install this package, please use the command
#  below, which will install all necessary build dependencies, compile
#  the package in an isolated environment, and then install it.
#  =====================================================================
#   $ pip install .
#  =====================================================================
#  If you are a software developer, and this is your own package, then
#  it is usually much more efficient to install the build dependencies
#  in your environment once and use the following command that avoids
#  a costly creation of a new virtual environment at every compilation:
#  =====================================================================
#   $ pip install nanobind scikit-build-core[pyproject]
#   $ pip install --no-build-isolation -ve .
#  =====================================================================
#  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
#  the package is imported. Otherwise, you need to re-run the above
#  after editing C++ files.")
#endif()


find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/ext/nanobind)

find_package(CUDAToolkit REQUIRED)


if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12 AND CUDAToolkit_VERSION_MINOR GREATER_EQUAL 2)
  message(STATUS "CUDA version is compatible (12.2 and above)")
else()
  message(FATAL_ERROR "CUDA version is not compatible (requires 12.2 and above)")
endif()

add_executable(_s2fft   
                ${CMAKE_CURRENT_LIST_DIR}/lib/src/extensions.cc
                ${CMAKE_CURRENT_LIST_DIR}/lib/src/s2fft.cu 
                ${CMAKE_CURRENT_LIST_DIR}/lib/src/plan_cache.cc 
                ${CMAKE_CURRENT_LIST_DIR}/lib/src/test.cc
              )

target_link_libraries(_s2fft PRIVATE CUDA::cudart CUDA::cufft)
target_include_directories(_s2fft PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include) 
set_target_properties(_s2fft PROPERTIES LINKER_LANGUAGE CXX)
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(_s2fft PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")


#nanobind_add_module(_s2fft STABLE_ABI  
#                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/plinalg.cc
#                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc 
#                    ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc.cu)
#
#target_include_directories(_s2fft
#PUBLIC
#${CMAKE_CURRENT_LIST_DIR}/lib/include
#${NVHPC_CUDA_INCLUDE_DIR}
#${MPI_CXX_INCLUDE_DIRS}
#${NVHPC_MATH_INCLUDE_DIR}
#)
#
#target_link_libraries(_s2fft PRIVATE MPI::MPI_CXX NVHPC::CUDA) 
#set_target_properties(_s2fft PROPERTIES LINKER_LANGUAGE CXX)
#
#install(TARGETS _s2fft LIBRARY DESTINATION plinalglib)
