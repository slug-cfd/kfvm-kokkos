cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::numeric")

add_library(numeric Numeric.cpp)
target_link_libraries(numeric Kokkos::kokkos)
target_compile_options(numeric PRIVATE -Wno-deprecated-gpu-targets)
target_include_directories(numeric PUBLIC ${USER_DIR})
target_include_directories(numeric PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_include_directories(numeric PUBLIC ${CMAKE_CURRENT_LIST_DIR})
