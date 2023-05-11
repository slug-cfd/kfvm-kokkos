#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#include "../../src/physics/EquationTypes.H"
#include "../../src/physics/RSTypes.H"
#include "../../src/numeric/RKTypes.H"

// Space dimension of problem
#define SPACE_DIM 2

// Stencil radius (2-4)
#define STENCIL_RADIUS 3

// Number of Gauss-Legendre points on faces (1-9)
#define NUM_QUAD_PTS 3

// Floating point precision (1 for double, 0 for single)
#define FLOAT_PRECISION 1

//#define KFVM_EXEC_SERIAL

constexpr KFVM::EquationType eqType = KFVM::EquationType::Hydro;
constexpr KFVM::RSType rsType = KFVM::RSType::HLLC;
constexpr KFVM::RKType rkType = KFVM::RKType::SSP4_3_2;

#endif
