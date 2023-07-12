#ifndef TYPES_H_
#define TYPES_H_

// File: Types.H
// Purpose: Define any types needed throughout the code base

#include <Kokkos_Core_fwd.hpp>
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

// 3 and 4 vectors
struct Vec3 {
  Real v1, v2, v3;
};

struct Vec4 {
  Real v1, v2, v3, v4;
};

// Enumeration of face names associated to [2*SPACE_DIM] entry in FaceDataView
enum FaceLabel { west = 0, east, south, north, bottom, top };

// Supported boundary conditions
enum BCType { periodic = 0, outflow, reflecting, user };

// Time stepper status codes
enum TSStatus { ACCEPTED = 0, TOLERANCE, UNPHYSICAL };

// Set number of evolved and auxiliary variables
constexpr std::size_t NUM_VARS = NUM_EQN_VARS[eqType];
constexpr std::size_t NUM_AUX = NUM_EQN_AUX[eqType];

// Kokkos spaces
#if (defined KFVM_EXEC_SERIAL)
typedef Kokkos::Serial ExecSpace;
typedef Kokkos::HostSpace MemSpace;
#elif (defined KFVM_EXEC_HOST)
typedef Kokkos::DefaultHostExecutionSpace ExecSpace;
typedef Kokkos::DefaultHostExecutionSpace::memory_space MemSpace;
#else
typedef Kokkos::DefaultExecutionSpace ExecSpace;
typedef Kokkos::DefaultExecutionSpace::memory_space MemSpace;
#endif
typedef Kokkos::SharedHostPinnedSpace MemSpacePinned;

// Kokkos views for various state data
#if (SPACE_DIM == 2)
typedef Kokkos::View<flag_t **, Kokkos::LayoutLeft, Kokkos::HostSpace> HostFlagDataView;
typedef Kokkos::View<Real **[2], MemSpace> WenoFlagView;
typedef Kokkos::View<Real **[2], Kokkos::LayoutLeft, Kokkos::HostSpace> HostWenoFlagView;
typedef Kokkos::View<Real **[NUM_VARS], MemSpace> ConsDataView;
typedef Kokkos::View<Real **[NUM_VARS], MemSpacePinned> ConsDataPinnedView;
typedef Kokkos::View<Real **[NUM_VARS], Kokkos::LayoutLeft, Kokkos::HostSpace>
    HostConsDataView;
typedef Kokkos::View<Real **[NUM_AUX], Kokkos::LayoutLeft, MemSpacePinned> AuxDataView;
typedef Kokkos::View<Real **[2][NUM_QUAD_PTS][NUM_VARS], MemSpace> FaceDataView;
typedef Kokkos::View<Real *[NUM_QUAD_PTS][NUM_VARS], MemSpacePinned> FaceDataPinnedView;
typedef Kokkos::View<Real **[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_VARS], MemSpace>
    SourceDataView;
#else
typedef Kokkos::View<flag_t ***, Kokkos::LayoutLeft, Kokkos::HostSpace> HostFlagDataView;
typedef Kokkos::View<Real ***[2], MemSpace> WenoFlagView;
typedef Kokkos::View<Real ***[2], Kokkos::LayoutLeft, Kokkos::HostSpace> HostWenoFlagView;
typedef Kokkos::View<Real ***[NUM_VARS], MemSpace> ConsDataView;
typedef Kokkos::View<Real ***[NUM_VARS], MemSpacePinned> ConsDataPinnedView;
typedef Kokkos::View<Real ***[NUM_VARS], Kokkos::LayoutLeft, Kokkos::HostSpace>
    HostConsDataView;
typedef Kokkos::View<Real ***[NUM_AUX], Kokkos::LayoutLeft, MemSpacePinned> AuxDataView;
typedef Kokkos::View<Real ***[2][NUM_QUAD_PTS * NUM_QUAD_PTS][NUM_VARS], MemSpace>
    FaceDataView;
typedef Kokkos::View<Real * * [NUM_QUAD_PTS * NUM_QUAD_PTS][NUM_VARS], MemSpacePinned>
    FaceDataPinnedView;
typedef Kokkos::View<Real ***[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_VARS],
                     MemSpace>
    SourceDataView;
#endif

// Kokkos views for quadrature rules and differentiation matrices
typedef Kokkos::View<Real[NUM_QUAD_PTS], MemSpace> QuadRuleView;
typedef Kokkos::View<Real[NUM_QUAD_PTS][NUM_QUAD_PTS + 2], MemSpace> DiffMatView;

// Unordered map for storing weno flags
typedef Kokkos::UnorderedMap<idx_t, void, MemSpace> FlagMap;
} // namespace KFVM

#endif
