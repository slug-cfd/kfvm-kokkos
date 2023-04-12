cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::io")

add_library(io WriterPDI.cpp)
target_include_directories(io PUBLIC ${USER_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(io PRIVATE -Wno-deprecated-gpu-targets)
