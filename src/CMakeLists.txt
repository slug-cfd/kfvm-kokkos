cmake_minimum_required(VERSION 3.16)

project("kfvm-kokkos::src")

add_subdirectory(numeric)
add_subdirectory(physics)
add_subdirectory(stencil)
add_subdirectory(io)

add_executable(kfvm.ex main.cpp ${USER_DIR}/ProblemSetup.cpp Solver.cpp BoundaryData.cpp ParameterStructs.cpp)

target_include_directories(kfvm.ex PUBLIC ${USER_DIR})

target_link_libraries(kfvm.ex numeric stencil io Kokkos::kokkoskernels Kokkos::kokkos PDI::PDI_C LAPACK::LAPACK MPI::MPI_CXX fmt::fmt CUDA::cuda_driver)

target_compile_options(kfvm.ex PRIVATE -Wno-deprecated-gpu-targets)
