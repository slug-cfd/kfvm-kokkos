cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::stencil")

add_library(stencil Stencil.cpp)
target_include_directories(stencil PUBLIC ${USER_DIR})
target_include_directories(stencil PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_include_directories(stencil PUBLIC ${CMAKE_CURRENT_LIST_DIR})
