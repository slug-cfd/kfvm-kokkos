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

template <>
struct RSPhys<EquationType::LinAdv> {
  using ConsVec = typename SimVar<EquationType::LinAdv>::ConsVec;
  using PrimVec = typename SimVar<EquationType::LinAdv>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    using Kokkos::cos;
    using Kokkos::sin;

    // Velocity on face
    const Real vx = cos(eosParams.gamma), vy = sin(eosParams.gamma);

    if (ord[0] == 0) {
      F[0] = vx * U[0];
    } else if (ord[0] == 1) {
      F[0] = vy * U[0];
    } else {
      F[0] = 0.0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  static Real speeds(const EosParameters &eosParams, const int ord[3], const ConsVec uL,
                     const ConsVec uR, const PrimVec vL, const PrimVec vR, Real &sL,
                     Real &sR) {
    using Kokkos::cos;
    using Kokkos::fabs;
    using Kokkos::sin;

    if (ord[0] == 0) {
      sL = cos(eosParams.gamma);
    } else if (ord[0] == 1) {
      sL = sin(eosParams.gamma);
    } else {
      sL = 0.0;
    }

    sR = sL;

    return fabs(sL);
  }
};

template <>
struct RiemannSolver<EquationType::LinAdv, RSType::EXACT> {
  using ConsVec = typename SimVar<EquationType::LinAdv>::ConsVec;
  using PrimVec = typename SimVar<EquationType::LinAdv>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static Real solve(const EosParameters &eosParams, const int dir, const ConsVec uL,
                    const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
    using Kokkos::cos;
    using Kokkos::fabs;
    using Kokkos::sin;

    // Velocity on face
    const Real vx = cos(eosParams.gamma), vy = sin(eosParams.gamma);

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
};

} // end namespace Physics

} // end namespace KFVM

#endif
