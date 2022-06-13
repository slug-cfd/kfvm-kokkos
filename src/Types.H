#ifndef TYPES_H_
#define TYPES_H_

// File: Types.H
// Purpose: Define any types needed throughout the code base

#include <Kokkos_Core.hpp>

#include "Dimension.H"
#include "SimVar.H"

namespace KFVM {

  // Set floating point precision for solver
#ifdef KFVM_SINGLE_PREC
  typedef float Real;
#else
  typedef double Real;
#endif

  // Kernel function types
  typedef std::function<std::complex<double>(std::complex<double>,double,double)> KernelFunc;
  
  // Array of fluid properties
  typedef std::array<Real,NUM_PROPS> FluidProperties;

  // Kokkos views for various state data
#if (SPACE_DIM == 2)
  typedef Kokkos::View<Real**[NUM_VARS]> CellDataView;
  typedef Kokkos::View<Real**[NUM_QUAD_PTS][NUM_VARS]> FaceDataView;
#else
  typedef Kokkos::View<Real***[NUM_VARS]> CellDataView;
  typedef Kokkos::View<Real***[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_VARS]> FaceDataView;
#endif
}

#endif
