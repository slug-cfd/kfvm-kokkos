#ifndef TYPES_H_
#define TYPES_H_

// File: Types.H
// Purpose: Define any types needed throughout the code base

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "SimVar.H"

namespace KFVM {

  // Set floating point precision for solver
#if (FLOAT_PRECISION == 0)
  typedef float Real;
#else
  typedef double Real;
#endif

  // Set index type for all iteration spaces
  typedef int64_t idx_t;

  // Array of fluid properties
  typedef std::array<Real,NUM_PROPS> FluidProperties;

  // Enumeration of face names associated to [2*SPACE_DIM] entry in FaceDataView
  enum FaceLabel {west = 0, east, south, north, bottom, top};

  // Kokkos spaces, mostly here for debugging
#ifdef KFVM_DEBUG
  typedef Kokkos::Serial ExecSpace;
  typedef Kokkos::HostSpace MemSpace;
#else
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef Kokkos::DefaultExecutionSpace::memory_space MemSpace;
#endif

  // Kokkos views for various state data
#if (SPACE_DIM == 2)
  typedef Kokkos::View<Real**[NUM_VARS],MemSpace> CellDataView;
  typedef Kokkos::View<Real**[NUM_AUX],MemSpace> AuxDataView;
  typedef Kokkos::View<Real**[2][NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataView;
#else
  typedef Kokkos::View<Real***[NUM_VARS],MemSpace> CellDataView;
  typedef Kokkos::View<Real***[NUM_AUX],MemSpace> AuxDataView;
  typedef Kokkos::View<Real***[2][NUM_QUAD_PTS*NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataView;
#endif

  // Kokkos views for quadrature rules
  typedef Kokkos::View<Real[NUM_QUAD_PTS],MemSpace> QuadRuleView;
}

#endif
