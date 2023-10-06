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

KOKKOS_INLINE_FUNCTION
void hydroFlux(Real gamma, int ord[3], const Hydro::ConsVec U, const Hydro::PrimVec V,
               Hydro::ConsVec F) {
  using Hydro::PrimVar;
  using Hydro::SimVar;

  F[SimVar::dens] = U[ord[0]];
  F[ord[0]] = U[ord[0]] * V[ord[0]] + V[PrimVar::pres];
  F[ord[1]] = U[ord[0]] * V[ord[1]];
  F[ord[2]] = U[ord[0]] * V[ord[2]];
  F[SimVar::etot] = V[ord[0]] * (U[SimVar::etot] + V[PrimVar::pres]);
}

// Wavespeed estimates
// See Batten et. al. https://doi.org/10.1137/S1064827593260140
KOKKOS_INLINE_FUNCTION
void hydroSpeed(Real gamma, int lng, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                const Hydro::PrimVec vL, const Hydro::PrimVec vR, Real &sL, Real &sR,
                Real &M) {
  using Hydro::PrimVar;
  using Hydro::SimVar;
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;

  // Roe estimates
  const Real hL = (uL[SimVar::etot] + vL[PrimVar::pres]) / uL[SimVar::dens],
             hR = (uR[SimVar::etot] + vR[PrimVar::pres]) / uR[SimVar::dens];
  const Real uTil = roeAvg(uL[SimVar::dens], uR[SimVar::dens], vL[lng], vR[lng]),
             hTil = roeAvg(uL[SimVar::dens], uR[SimVar::dens], hL, hR),
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

  sL = fmin(uL[lng] / uL[SimVar::dens] - aL, uTil - aTil);
  sR = fmax(uR[lng] / uR[SimVar::dens] + aR, uTil + aTil);
  M = fmax(vnL / aL, vnR / aR);
}

KOKKOS_INLINE_FUNCTION
Real hydroHLL(Real gamma, int ord[3], const Hydro::ConsVec uL, const Hydro::ConsVec uR,
              const Hydro::PrimVec vL, const Hydro::PrimVec vR, Real F[NUM_VARS]) {
  using Hydro::ConsVec;
  using Hydro::SimVar;
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
      F[SimVar::dens] = hll(sL, sR, F[SimVar::dens], FT[SimVar::dens], uL[SimVar::dens],
                            uR[SimVar::dens]);
      F[ord[0]] = hll(sL, sR, F[ord[0]], FT[ord[0]], uL[ord[0]], uR[ord[0]]);
      F[ord[1]] = hll(sL, sR, F[ord[1]], FT[ord[1]], uL[ord[1]], uR[ord[1]]);
      F[ord[2]] = hll(sL, sR, F[ord[2]], FT[ord[2]], uL[ord[2]], uR[ord[2]]);
      F[SimVar::etot] = hll(sL, sR, F[SimVar::etot], FT[SimVar::etot], uL[SimVar::etot],
                            uR[SimVar::etot]);
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
Real hydroHLLC(Real gamma, int ord[3], const Hydro::ConsVec uL, const Hydro::ConsVec uR,
               const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
  using Hydro::ConsVec;
  using Hydro::PrimVar;
  using Hydro::SimVar;
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
  const Real psiL = uL[SimVar::dens] * (sL - vL[ord[0]]),
             psiR = uR[SimVar::dens] * (sR - vR[ord[0]]),
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
    F[SimVar::dens] = sFacL * (sL * uL[SimVar::dens] - F[SimVar::dens]);
    F[ord[0]] = sFacL * (sL * uL[ord[0]] - F[ord[0]]) + pFacL;
    F[ord[1]] = sFacL * (sL * uL[ord[1]] - F[ord[1]]) + shFacL * vJump;
    F[ord[2]] = sFacL * (sL * uL[ord[2]] - F[ord[2]]) + shFacL * wJump;
    F[SimVar::etot] = sFacL * (sL * uL[SimVar::etot] - F[SimVar::etot]) + sS * pFacL;
  } else if (0 < sR) {
    hydroFlux(gamma, ord, uR, vR, F);
    F[SimVar::dens] = sFacR * (sR * uR[SimVar::dens] - F[SimVar::dens]);
    F[ord[0]] = sFacR * (sR * uR[ord[0]] - F[ord[0]]) + pFacR;
    F[ord[1]] = sFacR * (sR * uR[ord[1]] - F[ord[1]]) + shFacR * vJump;
    F[ord[2]] = sFacR * (sR * uR[ord[2]] - F[ord[2]]) + shFacR * wJump;
    F[SimVar::etot] = sFacR * (sR * uR[SimVar::etot] - F[SimVar::etot]) + sS * pFacR;
  } else {
    hydroFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

template <RSType rst>
struct HydroRS {};

template <>
struct HydroRS<RSType::HLL> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momx, SimVar::momy, SimVar::momz};
    return hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momy, SimVar::momz, SimVar::momx};
    return hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momz, SimVar::momx, SimVar::momy};
    return hydroHLL(gamma, ord, uL, uR, vL, vR, F);
  }
};

template <>
struct HydroRS<RSType::HLLC> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momx, SimVar::momy, SimVar::momz};
    return hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momy, SimVar::momz, SimVar::momx};
    return hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momz, SimVar::momx, SimVar::momy};
    return hydroHLLC(gamma, ord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
