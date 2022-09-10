cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::stencil")

add_library(stencil Stencil.cpp VectorValuedRA.cpp CSEKernel.cpp Monomials.cpp)
target_include_directories(stencil PUBLIC ${USER_DIR} ${CMAKE_CURRENT_LIST_DIR})
