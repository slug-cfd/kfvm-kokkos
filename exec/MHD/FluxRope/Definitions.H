#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#include "../../../src/numeric/RKTypes.H"
#include "../../../src/physics/EquationTypes.H"
#include "../../../src/physics/RSTypes.H"
#include "../../../src/stencil/KernelTypes.H"

// Space dimension of problem
#define SPACE_DIM 3

// Stencil radius (2-4)
#define STENCIL_RADIUS 2

// Number of Gauss-Legendre points on faces (1-5)
#define NUM_QUAD_PTS 2

// #define KFVM_EXEC_HOST

constexpr KFVM::KernelType kernelType = KFVM::KernelType::SE;
constexpr KFVM::EquationType eqType = KFVM::EquationType::MHD_GLM;
constexpr KFVM::RSType rsType = KFVM::RSType::MHD_GLM_KEPES;
constexpr KFVM::RKType rkType = KFVM::RKType::SSP4_3_2;

#endif
