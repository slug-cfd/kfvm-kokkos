#ifndef LINADV_RIEMANN_H_
#define LINADV_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "LinAdv_EOS.H"

namespace KFVM {

namespace Physics {

namespace LinAdv {
using ConsVec = typename SimVar<EquationType::LinAdv>::ConsVec;

KOKKOS_INLINE_FUNCTION
Real linadvExact(Real gamma, int dir, const ConsVec uL, const ConsVec uR,
                 Real F[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::fabs;
  using Kokkos::sin;

  // Velocity on face
  const Real vx = cos(gamma), vy = sin(gamma);

  if (dir == 0) {
    F[0] = vx > 0.0 ? vx * uL[0] : vx * uR[0];
    return fabs(vx);
  } else if (dir == 1) {
    F[0] = vy > 0.0 ? vy * uL[0] : vy * uR[0];
    return fabs(vy);
  } else {
    F[0] = 0.0;
  }

  return 0.0;
}

} // namespace LinAdv

template <RSType rst>
struct LinAdvRS {};

template <>
struct LinAdvRS<RSType::EXACT> {
  using ConsVec = typename SimVar<EquationType::LinAdv>::ConsVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, ConsVec F) {
    return LinAdv::linadvExact(gamma, 0, uL, uR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, ConsVec F) {
    return LinAdv::linadvExact(gamma, 1, uL, uR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, ConsVec F) {
    return LinAdv::linadvExact(gamma, 2, uL, uR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
