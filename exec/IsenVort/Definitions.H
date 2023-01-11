#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#include "../../src/physics/EquationTypes.H"
#include "../../src/numeric/RKTypes.H"

// Space dimension of problem
#define SPACE_DIM 2

// Stencil radius (2-3)
#define STENCIL_RADIUS 2

// Number of Gauss-Legendre points on faces (1-9)
#define NUM_QUAD_PTS 3

// Floating point precision (1 for double, 0 for single)
#define FLOAT_PRECISION 0

//#define KFVM_DEBUG

constexpr KFVM::EquationType eqType = KFVM::EquationType::Euler;
constexpr KFVM::RK_TYPE rkType = KFVM::RK_TYPE::SSP4_3_2;

#endif
