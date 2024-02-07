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

namespace SRHydro {
using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
using PrimVar = typename SimVar<EquationType::SRHydro>::PrimVar;
using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

KOKKOS_INLINE_FUNCTION
void srhydroFlux(const Real gamma, int ord[3], const ConsVec u, const PrimVec v,
                 ConsVec F) {
  F[ConsVar::dens] = u[ConsVar::dens] * v[ord[0]];
  F[ord[0]] = u[ord[0]] * v[ord[0]] + v[PrimVar::pres];
  F[ord[1]] = u[ord[1]] * v[ord[0]];
  F[ord[2]] = u[ord[2]] * v[ord[0]];
  F[ConsVar::etot] = u[ord[0]];
}

// Wavespeed estimates
KOKKOS_INLINE_FUNCTION
void srhydroSpeed(Real gamma, int lng, const PrimVec vL, const PrimVec vR, Real &sL,
                  Real &sR) {
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;

  // Get soundspeeds
  const Real aL = EOS<EquationType::SRHydro>::soundspeed(gamma, vL), aLsq = aL * aL;
  const Real aR = EOS<EquationType::SRHydro>::soundspeed(gamma, vR), aRsq = aR * aR;

  // Left state intermediate quantities
  const Real wL = vL[PrimVar::lorz], sigL = aLsq / (wL * wL * (Real(1.0) - aLsq));
  const Real sqrtL = sqrt(sigL * (Real(1.0) + sigL - vL[lng] * vL[lng]));

  // Right state intermediate quantities
  const Real wR = vR[PrimVar::lorz], sigR = aRsq / (wR * wR * (Real(1.0) - aRsq));
  const Real sqrtR = sqrt(sigR * (Real(1.0) + sigR - vR[lng] * vR[lng]));

  // Left going wavespeeds
  const Real sLm = (vL[lng] - sqrtL) / (Real(1.0) + sigL);
  const Real sRm = (vR[lng] - sqrtR) / (Real(1.0) + sigR);

  // Right going wavespeeds
  const Real sLp = (vL[lng] + sqrtL) / (Real(1.0) + sigL);
  const Real sRp = (vR[lng] + sqrtR) / (Real(1.0) + sigR);

  // Put it all together
  sL = fmin(sLm, sRm);
  sR = fmax(sLp, sRp);
}

KOKKOS_INLINE_FUNCTION
Real srhydroLLF(Real gamma, int ord[3], const ConsVec uL, const ConsVec uR,
                const PrimVec vL, const PrimVec vR, ConsVec F) {
  using Kokkos::fabs;
  using Kokkos::fmax;

  // Wave speed estimates
  Real sL, sR, sMax;
  srhydroSpeed(gamma, ord[0], vL, vR, sL, sR);
  sMax = fmax(fabs(sL), fabs(sR));

  // Evaluate left and right fluxes
  ConsVec FT;
  srhydroFlux(gamma, ord, uL, vL, F);
  srhydroFlux(gamma, ord, uR, vR, FT);

  // Combine into LLF flux
  for (int nV = 0; nV < NUM_VARS; nV++) {
    F[nV] = Real(0.5) * ((F[nV] + FT[nV]) - sMax * (uR[nV] - uL[nV]));
  }

  return sMax;
}

KOKKOS_INLINE_FUNCTION
Real srhydroHLL(Real gamma, int ord[3], const ConsVec uL, const ConsVec uR,
                const PrimVec vL, const PrimVec vR, ConsVec F) {
  using Kokkos::fabs;
  using Kokkos::fmax;

  ConsVec FT;
  Real sL, sR;

  // Wave speed estimates
  srhydroSpeed(gamma, ord[0], vL, vR, sL, sR);

  // Find appropriate region and fill in flux vector
  if (sR > 0) {
    srhydroFlux(gamma, ord, uL, vL, F);
    if (sL <= 0) {
      srhydroFlux(gamma, ord, uR, vR, FT);
      F[ConsVar::dens] = hll(sL, sR, F[ConsVar::dens], FT[ConsVar::dens],
                             uL[ConsVar::dens], uR[ConsVar::dens]);
      F[ord[0]] = hll(sL, sR, F[ord[0]], FT[ord[0]], uL[ord[0]], uR[ord[0]]);
      F[ord[1]] = hll(sL, sR, F[ord[1]], FT[ord[1]], uL[ord[1]], uR[ord[1]]);
      F[ord[2]] = hll(sL, sR, F[ord[2]], FT[ord[2]], uL[ord[2]], uR[ord[2]]);
      F[ConsVar::etot] = hll(sL, sR, F[ConsVar::etot], FT[ConsVar::etot],
                             uL[ConsVar::etot], uR[ConsVar::etot]);
    }
  } else {
    srhydroFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

} // namespace SRHydro

template <RSType rst>
struct SRHydroRS {};

template <>
struct SRHydroRS<RSType::LLF> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    return SRHydro::srhydroLLF(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    return SRHydro::srhydroLLF(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    return SRHydro::srhydroLLF(gamma, ord, uL, uR, vL, vR, F);
  }
};

template <>
struct SRHydroRS<RSType::HLL> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    return SRHydro::srhydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    return SRHydro::srhydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    return SRHydro::srhydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
