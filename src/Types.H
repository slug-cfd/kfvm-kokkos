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

  // Enumeration of face names associated to [2*SPACE_DIM] entry in FaceDataView
  enum FaceLabel {west = 0, east, south, north, bottom, top};

  // Kokkos views for various state data
#if (SPACE_DIM == 2)
  typedef Kokkos::View<Real**[NUM_VARS]> CellDataView;
  typedef Kokkos::View<Real**[2*SPACE_DIM][NUM_QUAD_PTS][NUM_VARS]> FaceDataView;
#else
  typedef Kokkos::View<Real***[NUM_VARS]> CellDataView;
  typedef Kokkos::View<Real***[2*SPACE_DIM][NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_VARS]> FaceDataView;
#endif
}

#endif
