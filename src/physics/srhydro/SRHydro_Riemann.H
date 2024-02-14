#ifndef SRHYDRO_RIEMANN_H_
#define SRHYDRO_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "SRHydro_EOS.H"

namespace KFVM {

namespace Physics {

template <>
struct RSPhys<EquationType::SRHydro> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::SRHydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    (void)eosParams;
    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};

    F[ConsVar::dens] = U[ConsVar::dens] * V[vord[0]];
    F[vord[0]] = U[vord[0]] * V[vord[0]] + V[PrimVar::pres];
    F[vord[1]] = U[vord[1]] * V[vord[0]];
    F[vord[2]] = U[vord[2]] * V[vord[0]];
    F[ConsVar::etot] = U[vord[0]];
  }

  // Wavespeed estimates
  KOKKOS_INLINE_FUNCTION
  static Real speeds(const EosParameters &eosParams, const int ord[3], const ConsVec uL,
                     const ConsVec uR, const PrimVec vL, const PrimVec vR, Real &sL,
                     Real &sR) {
    using Kokkos::fmax;
    using Kokkos::fmin;
    using Kokkos::sqrt;

    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};
    const Real gamma = eosParams.gamma;

    // Get soundspeeds
    const Real aL = EOS<EquationType::SRHydro>::soundspeed(gamma, vL), aLsq = aL * aL;
    const Real aR = EOS<EquationType::SRHydro>::soundspeed(gamma, vR), aRsq = aR * aR;

    // Left state intermediate quantities
    const Real wL = vL[PrimVar::lorz], sigL = aLsq / (wL * wL * (Real(1.0) - aLsq));
    const Real sqrtL = sqrt(sigL * (Real(1.0) + sigL - vL[vord[0]] * vL[vord[0]]));

    // Right state intermediate quantities
    const Real wR = vR[PrimVar::lorz], sigR = aRsq / (wR * wR * (Real(1.0) - aRsq));
    const Real sqrtR = sqrt(sigR * (Real(1.0) + sigR - vR[vord[0]] * vR[vord[0]]));

    // Left going wavespeeds
    const Real sLm = (vL[vord[0]] - sqrtL) / (Real(1.0) + sigL);
    const Real sRm = (vR[vord[0]] - sqrtR) / (Real(1.0) + sigR);

    // Right going wavespeeds
    const Real sLp = (vL[vord[0]] + sqrtL) / (Real(1.0) + sigL);
    const Real sRp = (vR[vord[0]] + sqrtR) / (Real(1.0) + sigR);

    // Put it all together
    sL = fmin(sLm, sRm);
    sR = fmax(sLp, sRp);

    return fmax(fabs(sL), fabs(sR));
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
