cmake_minimum_required (VERSION 3.16)

project ("kfvm-kokkos")

set (KFVM_ROOT ${CMAKE_CURRENT_LIST_DIR})

set (CMAKE_MODULE_PATH ${KFVM_ROOT}/CMakeModules)

set (CMAKE_CXX_EXTENSIONS OFF)

find_package (Kokkos REQUIRED)
find_package (KokkosKernels REQUIRED)
find_package (cerf REQUIRED)
find_package (NetCDF REQUIRED)
find_package (LAPACK REQUIRED)

# If only I knew how to write a FindJUNK.cmake file...
set (CMAKE_PREFIX_PATH /usr/local/lib/cmake/netCDF)
find_package (netCDFCxx REQUIRED)

message (${CMAKE_CXX_COMPILER})

set (USER_DIR ${CMAKE_BINARY_DIR})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${USER_DIR}/bin)

add_subdirectory (src)
