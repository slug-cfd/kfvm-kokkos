cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src::physics")

add_subdirectory(hydro)
add_subdirectory(navierstokes)
add_subdirectory(mhd)
add_subdirectory(mhd_8w)
add_subdirectory(srhydro)
add_subdirectory(linadv)
