cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::stencil")

add_library(stencil Stencil.cpp ReconVectors.cpp StencilSymmetry.cpp Monomials.cpp)
target_link_libraries(stencil Kokkos::kokkoskernels Kokkos::kokkos LAPACK::LAPACK)
target_include_directories(stencil PUBLIC ${USER_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(stencil PRIVATE -Wno-deprecated-gpu-targets)
