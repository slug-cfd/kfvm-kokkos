#ifndef TYPES_H_
#define TYPES_H_

// File: Types.H
// Purpose: Define any types needed throughout the code base

#include <cstddef>
#include <cstdint>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "physics/EquationTypes.H"

namespace KFVM {

  // Set floating point precision for solver
#if (FLOAT_PRECISION == 0)
  typedef float Real;
#error Single precision temporarily unsupported.
#else
  typedef double Real;
#endif

  // Set index type for all iteration spaces
  typedef int64_t idx_t;

  // Type for small flags
  typedef uint8_t flag_t;

  // Enumeration of face names associated to [2*SPACE_DIM] entry in FaceDataView
  enum FaceLabel {west = 0,east,south,north,bottom,top};
  
  // Supported boundary conditions
  enum BCType {periodic = 0,outflow,reflecting,user};

  // Time stepper status codes
  enum TSStatus {ACCEPTED = 0,TOLERANCE,UNPHYSICAL};

  // Weno selection flags
  enum WenoFlag {NOWENO = 0,NBRWENO,WENO};

  // Set number of evolved and auxiliary variables
  constexpr std::size_t NUM_VARS = NUM_EQN_VARS[eqType];
  constexpr std::size_t NUM_AUX = NUM_EQN_AUX[eqType];

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
  typedef Kokkos::View<flag_t**,Kokkos::LayoutLeft,Kokkos::HostSpace> HostFlagDataView;
  typedef Kokkos::View<Real**,MemSpace> CellDataView;
  typedef Kokkos::View<Real**,Kokkos::LayoutLeft,Kokkos::HostSpace> HostCellDataView;
  typedef Kokkos::View<Real**[NUM_VARS],MemSpace> ConsDataView;
  typedef Kokkos::View<Real**[NUM_VARS],MemSpace> ConsDataCommView;
  typedef Kokkos::View<Real**[NUM_VARS],Kokkos::LayoutLeft,Kokkos::HostSpace> HostConsDataView;
  typedef Kokkos::View<Real**[NUM_AUX],MemSpace> AuxDataView;
  typedef Kokkos::View<Real**[NUM_AUX],Kokkos::LayoutLeft,Kokkos::HostSpace> HostAuxDataView;
  typedef Kokkos::View<Real**[2][NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataView;
  typedef Kokkos::View<Real*[NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataCommView;
#else
  typedef Kokkos::View<flag_t***,Kokkos::LayoutLeft,Kokkos::HostSpace> HostFlagDataView;
  typedef Kokkos::View<Real***,MemSpace> CellDataView;
  typedef Kokkos::View<Real***,Kokkos::LayoutLeft,Kokkos::HostSpace> HostCellDataView;
  typedef Kokkos::View<Real***[NUM_VARS],MemSpace> ConsDataView;
  typedef Kokkos::View<Real***[NUM_VARS],MemSpace> ConsDataCommView;
  typedef Kokkos::View<Real***[NUM_VARS],Kokkos::LayoutLeft,Kokkos::HostSpace> HostConsDataView;
  typedef Kokkos::View<Real***[NUM_AUX],MemSpace> AuxDataView;
  typedef Kokkos::View<Real***[NUM_AUX],Kokkos::LayoutLeft,Kokkos::HostSpace> HostAuxDataView;
  typedef Kokkos::View<Real***[2][NUM_QUAD_PTS*NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataView;
  typedef Kokkos::View<Real**[NUM_QUAD_PTS*NUM_QUAD_PTS][NUM_VARS],MemSpace> FaceDataCommView;
#endif

  // Kokkos views for quadrature rules
  typedef Kokkos::View<Real[NUM_QUAD_PTS],MemSpace> QuadRuleView;

  // Unordered map for storing weno flags
  typedef Kokkos::UnorderedMap<idx_t,void,ExecSpace> FlagMap;
}

#endif
