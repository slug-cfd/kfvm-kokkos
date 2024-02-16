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

template <>
struct RSPhys<EquationType::Hydro> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::Hydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    (void)eosParams;
    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};

    F[ConsVar::dens] = U[vord[0]];
    F[vord[0]] = U[vord[0]] * V[vord[0]] + V[PrimVar::pres];
    F[vord[1]] = U[vord[0]] * V[vord[1]];
    F[vord[2]] = U[vord[0]] * V[vord[2]];
    F[ConsVar::etot] = V[vord[0]] * (U[ConsVar::etot] + V[PrimVar::pres]);
  }

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   ConsVec F) {
    PrimVec V;
    EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, U, V);
    flux(eosParams, ord, U, V, F);
  }

  // Wavespeed estimates
  // See Batten et. al. https://doi.org/10.1137/S1064827593260140
  KOKKOS_INLINE_FUNCTION
  static Real speeds(const EosParameters &eosParams, const int ord[3], const ConsVec uL,
                     const ConsVec uR, const PrimVec vL, const PrimVec vR, Real &sL,
                     Real &sR) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;
    using Kokkos::sqrt;

    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};

    // Useful left and right quantities
    const Real rhoL = uL[ConsVar::dens], rhoR = uR[ConsVar::dens];
    const Real hL = (uL[ConsVar::etot] + vL[PrimVar::pres]) / rhoL,
               hR = (uR[ConsVar::etot] + vR[PrimVar::pres]) / rhoR;
    const Real aL = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, uL),
               aR = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, uR);

    // Roe estimates
    const Real uTil = roeAvg(rhoL, rhoR, vL[vord[0]], vR[vord[0]]),
               vTil = roeAvg(rhoL, rhoR, vL[vord[1]], vR[vord[1]]),
               wTil = roeAvg(rhoL, rhoR, vL[vord[2]], vR[vord[2]]),
               vsqTil = uTil * uTil + vTil * vTil + wTil * wTil,
               hTil = roeAvg(rhoL, rhoR, hL, hR),
               aTil = sqrt((eosParams.gamma - 1.0) * (hTil - 0.5 * vsqTil));

    // Batten estimate
    sL = fmin(vL[vord[0]] - aL, uTil - aTil);
    sR = fmax(vR[vord[0]] + aR, uTil + aTil);

    return fmax(fabs(sL), fabs(sR));
  }

  KOKKOS_INLINE_FUNCTION
  static Real speed(const EosParameters &eosParams, const int ord[3], const ConsVec u) {
    const int vlng = ConsVar::momx + ord[0];

    const Real a = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, u);

    return fabs(u[vlng] / u[ConsVar::dens]) + a;
  }
};

// This incorporates the low-mach fix from the HLLC+ method
// It does *not* do any comparisons with neighboring cells (fM variable),
// and does not incorporate the shear velocity fix for carbuncling
// see https://doi.org/10.1137/18M119032X
template <>
struct RiemannSolver<EquationType::Hydro, RSType::HLLC> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::Hydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static Real solve(const EosParameters &eosParams, const int dir, const ConsVec uL,
                    const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;
    using Kokkos::pow;
    using Kokkos::sqrt;

    // set field orderings to collapse to 1d solver
    const int ord[] = {dir, (dir + 1) % 3, (dir + 2) % 3};
    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};

    // Wave speed estimates
    Real sL, sR, sMax;
    sMax = RSPhys<EquationType::Hydro>::speeds(eosParams, ord, uL, uR, vL, vR, sL, sR);

    // Mach estimate
    const Real vsqL = vL[PrimVar::velx] * vL[PrimVar::velx] +
                      vL[PrimVar::vely] * vL[PrimVar::vely] +
                      vL[PrimVar::velz] * vL[PrimVar::velz];
    const Real vsqR = vR[PrimVar::velx] * vR[PrimVar::velx] +
                      vR[PrimVar::vely] * vR[PrimVar::vely] +
                      vR[PrimVar::velz] * vR[PrimVar::velz];
    const Real aL = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, uL),
               aR = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, uR);
    Real M = fmax(sqrt(vsqL) / aL, sqrt(vsqR) / aR);

    // Star region speed and pressure
    const Real sS = hllcSStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                              vR[PrimVar::pres], vL[vord[0]], vR[vord[0]], sL, sR);
    const Real pS = hllcPStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                              vR[PrimVar::pres], vL[vord[0]], vR[vord[0]], sL, sR, sS);

    // Wavespeed prefactors
    const Real sFacL = sS / (sL - sS), sFacR = sS / (sR - sS);

    // Velocity jumps
    const Real uJump = vR[vord[0]] - vL[vord[0]], vJump = vR[vord[1]] - vL[vord[1]],
               wJump = vR[vord[2]] - vL[vord[2]];

    // Psi terms (weighted speed differences)
    const Real psiL = uL[ConsVar::dens] * (sL - vL[vord[0]]),
               psiR = uR[ConsVar::dens] * (sR - vR[vord[0]]),
               psiFac = psiR * psiL / (psiR - psiL);

    // Shear fix
    const Real presRat = fmin(vL[PrimVar::pres] / vR[PrimVar::pres],
                              vR[PrimVar::pres] / vL[PrimVar::pres]);
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
      RSPhys<EquationType::Hydro>::flux(eosParams, ord, uL, vL, F);
    } else if (0 <= sS) {
      RSPhys<EquationType::Hydro>::flux(eosParams, ord, uL, vL, F);
      F[ConsVar::dens] = sFacL * (sL * uL[ConsVar::dens] - F[ConsVar::dens]);
      F[vord[0]] = sFacL * (sL * uL[vord[0]] - F[vord[0]]) + pFacL;
      F[vord[1]] = sFacL * (sL * uL[vord[1]] - F[vord[1]]) + shFacL * vJump;
      F[vord[2]] = sFacL * (sL * uL[vord[2]] - F[vord[2]]) + shFacL * wJump;
      F[ConsVar::etot] = sFacL * (sL * uL[ConsVar::etot] - F[ConsVar::etot]) + sS * pFacL;
    } else if (0 < sR) {
      RSPhys<EquationType::Hydro>::flux(eosParams, ord, uR, vR, F);
      F[ConsVar::dens] = sFacR * (sR * uR[ConsVar::dens] - F[ConsVar::dens]);
      F[vord[0]] = sFacR * (sR * uR[vord[0]] - F[vord[0]]) + pFacR;
      F[vord[1]] = sFacR * (sR * uR[vord[1]] - F[vord[1]]) + shFacR * vJump;
      F[vord[2]] = sFacR * (sR * uR[vord[2]] - F[vord[2]]) + shFacR * wJump;
      F[ConsVar::etot] = sFacR * (sR * uR[ConsVar::etot] - F[ConsVar::etot]) + sS * pFacR;
    } else {
      RSPhys<EquationType::Hydro>::flux(eosParams, ord, uR, vR, F);
    }

    return sMax;
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
