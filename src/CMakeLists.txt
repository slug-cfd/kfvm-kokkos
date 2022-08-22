cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src")

add_subdirectory(numeric)
add_subdirectory(hydro)
add_subdirectory(stencil)

add_executable(kfvm.ex main.cpp ${USER_DIR}/ProblemSetup.cpp NetCDFWriter.cpp Solver.cpp)
target_include_directories(kfvm.ex PUBLIC ${USER_DIR} ${NetCDF_INCLUDE_DIRS})
target_link_libraries(kfvm.ex numeric stencil Kokkos::kokkoskernels Kokkos::kokkos NetCDF::NetCDF netCDF::netcdf-cxx4 LAPACK::LAPACK)
