#ifndef EQUATIONTYPES_K_H_
#define EQUATIONTYPES_K_H_

#include <array>

namespace KFVM {
  
  enum EquationType { Hydro = 0, MHD_GLM, SRHydro, NUM_EQUATION_TYPES };

  constexpr std::array<std::size_t,NUM_EQUATION_TYPES> NUM_EQN_VARS {{5,9,5}};
  constexpr std::array<std::size_t,NUM_EQUATION_TYPES> NUM_EQN_AUX {{5,7,5}};
  
} // end namespace KFVM

#endif
