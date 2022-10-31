#include <array>

#include "ProblemSetup.H"
#include "BoundaryConditions_K.H"

// Names of variables inside solution files
std::array<std::string,NUM_VARS> KFVM::ProblemSetup::varName = {
  "dens",
  "momx",
  "momy",
  "momz",
  "etot"};

std::array<std::string,NUM_AUX> KFVM::ProblemSetup::auxVarName = {
  "velx",
  "vely",
  "velz",
  "eint",
  "pres"};

// Boundary conditions in order west, east, south, north, bottom, top
// Note: Always all 6, even in 2D. bottom/top ignored in 2D
std::array<KFVM::BoundaryConditions::BCType,6> KFVM::ProblemSetup::bcType = {
  KFVM::BoundaryConditions::BCType::periodic,
  KFVM::BoundaryConditions::BCType::periodic,
  KFVM::BoundaryConditions::BCType::periodic,
  KFVM::BoundaryConditions::BCType::periodic,
  KFVM::BoundaryConditions::BCType::periodic,
  KFVM::BoundaryConditions::BCType::periodic};
