#include <array>

#include "ProblemSetup.H"
#include "BoundaryConditions_K.H"
#include "physics/EquationTypes.H"

// Names of variables inside solution files
std::array<std::string, KFVM::NUM_VARS> KFVM::ProblemSetup::varName = {
  "dens",
  "momx",
  "momy",
  "momz",
  "etot",
  "magx",
  "magy",
  "magz",
  "psi"};

std::array<std::string, KFVM::NUM_AUX> KFVM::ProblemSetup::auxVarName = {
  "velx",
  "vely",
  "velz",
  "eint",
  "pres",
  "prsg",
  "prsb"};

// Boundary conditions in order west, east, south, north, bottom, top
// Note: Always all 6, even in 2D. bottom/top ignored in 2D
std::array<KFVM::BCType,6> KFVM::ProblemSetup::bcType = {
  KFVM::BCType::outflow,
  KFVM::BCType::outflow,
  KFVM::BCType::user,
  KFVM::BCType::outflow,
  KFVM::BCType::reflecting,
  KFVM::BCType::outflow};

// Set whether this problem has source terms
bool KFVM::ProblemSetup::haveSourceTerms = true;
