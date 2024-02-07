#ifndef HYDRO_RIEMANN_H_
#define HYDRO_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "Hydro_EOS.H"

namespace KFVM {

namespace Physics {

namespace Hydro {
using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
using PrimVar = typename SimVar<EquationType::Hydro>::PrimVar;
using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

KOKKOS_INLINE_FUNCTION
void hydroFlux(Real gamma, int ord[3], const ConsVec U, const PrimVec V, ConsVec F) {
  F[ConsVar::dens] = U[ord[0]];
  F[ord[0]] = U[ord[0]] * V[ord[0]] + V[PrimVar::pres];
  F[ord[1]] = U[ord[0]] * V[ord[1]];
  F[ord[2]] = U[ord[0]] * V[ord[2]];
  F[ConsVar::etot] = V[ord[0]] * (U[ConsVar::etot] + V[PrimVar::pres]);
}

// Wavespeed estimates
// See Batten et. al. https://doi.org/10.1137/S1064827593260140
KOKKOS_INLINE_FUNCTION
void hydroSpeed(Real gamma, int lng, const ConsVec uL, const ConsVec uR, const PrimVec vL,
                const PrimVec vR, Real &sL, Real &sR, Real &M) {
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;

  // Roe estimates
  const Real hL = (uL[ConsVar::etot] + vL[PrimVar::pres]) / uL[ConsVar::dens],
             hR = (uR[ConsVar::etot] + vR[PrimVar::pres]) / uR[ConsVar::dens];
  const Real uTil = roeAvg(uL[ConsVar::dens], uR[ConsVar::dens], vL[lng], vR[lng]),
             hTil = roeAvg(uL[ConsVar::dens], uR[ConsVar::dens], hL, hR),
             aTil = sqrt((gamma - 1.0) * (hTil - 0.5 * uTil * uTil));

  // Batten estimate
  const Real aL = EOS<EquationType::Hydro>::soundspeed(gamma, uL),
             aR = EOS<EquationType::Hydro>::soundspeed(gamma, uR);

  // Velocity magnitudes for Mach estimate
  const Real vnL =
      sqrt(vL[PrimVar::velx] * vL[PrimVar::velx] + vL[PrimVar::vely] * vL[PrimVar::vely] +
           vL[PrimVar::velz] * vL[PrimVar::velz]);
  const Real vnR =
      sqrt(vR[PrimVar::velx] * vR[PrimVar::velx] + vR[PrimVar::vely] * vR[PrimVar::vely] +
           vR[PrimVar::velz] * vR[PrimVar::velz]);

  sL = fmin(uL[lng] / uL[ConsVar::dens] - aL, uTil - aTil);
  sR = fmax(uR[lng] / uR[ConsVar::dens] + aR, uTil + aTil);
  M = fmax(vnL / aL, vnR / aR);
}

KOKKOS_INLINE_FUNCTION
Real hydroHLL(Real gamma, int ord[3], const ConsVec uL, const ConsVec uR,
              const PrimVec vL, const PrimVec vR, Real F[NUM_VARS]) {
  using Kokkos::fabs;
  using Kokkos::fmax;

  ConsVec FT;
  Real sL, sR, M;

  // Wave speed estimates
  hydroSpeed(gamma, ord[0], uL, uR, vL, vR, sL, sR, M);

  // Find appropriate region and fill in flux vector
  if (sR > 0) {
    hydroFlux(gamma, ord, uL, vL, F);
    if (sL <= 0) {
      hydroFlux(gamma, ord, uR, vR, FT);
      F[ConsVar::dens] = hll(sL, sR, F[ConsVar::dens], FT[ConsVar::dens],
                             uL[ConsVar::dens], uR[ConsVar::dens]);
      F[ord[0]] = hll(sL, sR, F[ord[0]], FT[ord[0]], uL[ord[0]], uR[ord[0]]);
      F[ord[1]] = hll(sL, sR, F[ord[1]], FT[ord[1]], uL[ord[1]], uR[ord[1]]);
      F[ord[2]] = hll(sL, sR, F[ord[2]], FT[ord[2]], uL[ord[2]], uR[ord[2]]);
      F[ConsVar::etot] = hll(sL, sR, F[ConsVar::etot], FT[ConsVar::etot],
                             uL[ConsVar::etot], uR[ConsVar::etot]);
    }
  } else {
    hydroFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

// This incorporates the low-mach fix from the HLLC+ method
// It does *not* do any comparisons with neighboring cells (fM variable),
// and does not incorporate the shear velocity fix for carbuncling
// see https://doi.org/10.1137/18M119032X
KOKKOS_INLINE_FUNCTION
Real hydroHLLC(Real gamma, int ord[3], const ConsVec uL, const ConsVec uR,
               const PrimVec vL, const PrimVec vR, ConsVec F) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::pow;
  using Kokkos::sqrt;

  // Wave speed estimates
  Real sL, sR, M;
  hydroSpeed(gamma, ord[0], uL, uR, vL, vR, sL, sR, M);

  // Star region speed and pressure
  const Real sS = hllcSStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                            vR[PrimVar::pres], vL[ord[0]], vR[ord[0]], sL, sR);
  const Real pS = hllcPStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                            vR[PrimVar::pres], vL[ord[0]], vR[ord[0]], sL, sR, sS);

  // Wavespeed prefactors
  const Real sFacL = sS / (sL - sS), sFacR = sS / (sR - sS);

  // Velocity jumps
  const Real uJump = vR[ord[0]] - vL[ord[0]], vJump = vR[ord[1]] - vL[ord[1]],
             wJump = vR[ord[2]] - vL[ord[2]];

  // Psi terms (weighted speed differences)
  const Real psiL = uL[ConsVar::dens] * (sL - vL[ord[0]]),
             psiR = uR[ConsVar::dens] * (sR - vR[ord[0]]),
             psiFac = psiR * psiL / (psiR - psiL);

  // Shear fix
  const Real presRat =
      fmin(vL[PrimVar::pres] / vR[PrimVar::pres], vR[PrimVar::pres] / vL[PrimVar::pres]);
  const Real gFac = 1.0 - pow(presRat, M);
  const Real shFacL = psiFac * gFac * sL / (sL - sS),
             shFacR = psiFac * gFac * sR / (sR - sS);

  // Low mach fix
  M = fmin(M, 1.0); // Everything beyond mach 1 is the same
  const Real fM = M * sqrt(4.0 + pow(1.0 - M * M, 2)) / (1.0 + M * M);
  const Real pFacL = psiFac * (fM - 1.0) * uJump + sL * pS / (sL - sS);
  const Real pFacR = psiFac * (fM - 1.0) * uJump + sR * pS / (sR - sS);

  // Find correct zone
  if (0 <= sL) {
    hydroFlux(gamma, ord, uL, vL, F);
  } else if (0 <= sS) {
    hydroFlux(gamma, ord, uL, vL, F);
    F[ConsVar::dens] = sFacL * (sL * uL[ConsVar::dens] - F[ConsVar::dens]);
    F[ord[0]] = sFacL * (sL * uL[ord[0]] - F[ord[0]]) + pFacL;
    F[ord[1]] = sFacL * (sL * uL[ord[1]] - F[ord[1]]) + shFacL * vJump;
    F[ord[2]] = sFacL * (sL * uL[ord[2]] - F[ord[2]]) + shFacL * wJump;
    F[ConsVar::etot] = sFacL * (sL * uL[ConsVar::etot] - F[ConsVar::etot]) + sS * pFacL;
  } else if (0 < sR) {
    hydroFlux(gamma, ord, uR, vR, F);
    F[ConsVar::dens] = sFacR * (sR * uR[ConsVar::dens] - F[ConsVar::dens]);
    F[ord[0]] = sFacR * (sR * uR[ord[0]] - F[ord[0]]) + pFacR;
    F[ord[1]] = sFacR * (sR * uR[ord[1]] - F[ord[1]]) + shFacR * vJump;
    F[ord[2]] = sFacR * (sR * uR[ord[2]] - F[ord[2]]) + shFacR * wJump;
    F[ConsVar::etot] = sFacR * (sR * uR[ConsVar::etot] - F[ConsVar::etot]) + sS * pFacR;
  } else {
    hydroFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

} // namespace Hydro

template <RSType rst>
struct HydroRS {};

template <>
struct HydroRS<RSType::HLL> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    return Hydro::hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    return Hydro::hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    return Hydro::hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }
};

template <>
struct HydroRS<RSType::HLLC> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    return Hydro::hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    return Hydro::hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int ord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    return Hydro::hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
