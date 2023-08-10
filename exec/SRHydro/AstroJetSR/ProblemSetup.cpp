#include <array>

#include "BoundaryConditions_K.H"
#include "ProblemSetup.H"

// Boundary conditions in order west, east, south, north, bottom, top
// Note: Always all 6, even in 2D. bottom/top ignored in 2D
std::array<KFVM::BCType, 6> KFVM::ProblemSetup::bcType = {
    KFVM::BCType::user,    KFVM::BCType::outflow,    KFVM::BCType::reflecting,
    KFVM::BCType::outflow, KFVM::BCType::reflecting, KFVM::BCType::outflow};

// Set whether this problem has source terms
bool KFVM::ProblemSetup::haveSourceTerms = true;
