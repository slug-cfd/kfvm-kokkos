#ifndef EQUATIONTYPES_K_H_
#define EQUATIONTYPES_K_H_

#include <array>

namespace KFVM {

enum EquationType {
  Hydro = 0,
  NavierStokes,
  MHD_GLM,
  MHD_8W,
  SRHydro,
  LinAdv,
  NUM_EQUATION_TYPES
};

constexpr std::array<std::size_t, NUM_EQUATION_TYPES> NUM_EQN_VARS{{5, 5, 9, 8, 5, 1}};
constexpr std::array<std::size_t, NUM_EQUATION_TYPES> NUM_EQN_AUX{{5, 5, 6, 6, 6, 1}};
constexpr std::array<std::size_t, NUM_EQUATION_TYPES> NUM_EQN_STATS{{2, 2, 3, 3, 1, 1}};

} // end namespace KFVM

#endif
