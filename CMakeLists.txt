cmake_minimum_required (VERSION 3.16)

project ("kfvm-kokkos")

enable_language (CXX CUDA)

set (CMAKE_CXX_STANDARD 17)

set (KFVM_ROOT ${CMAKE_CURRENT_LIST_DIR})

set (CMAKE_CXX_EXTENSIONS OFF)

find_package (MPI REQUIRED)
find_package (Kokkos REQUIRED)
find_package (KokkosKernels REQUIRED)
find_package (PDI REQUIRED)
find_package (LAPACK REQUIRED)
find_package (fmt REQUIRED)

# This fixes missing CUDA::cuda_driver bug
# This also hard-locks the code into needing cuda...
find_package (CUDAToolkit REQUIRED)

set (USER_DIR ${CMAKE_BINARY_DIR})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${USER_DIR}/bin)

# Fixes for Lux
if (NOT ${CMAKE_PREFIX_PATH} STREQUAL "")
include_directories (SYSTEM ${MPI_INCLUDE_PATH})
include_directories (SYSTEM ${CMAKE_PREFIX_PATH}/include)
endif ()

add_subdirectory (src)
