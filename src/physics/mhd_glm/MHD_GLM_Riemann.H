#ifndef MHD_GLM_RIEMANN_H_
#define MHD_GLM_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "MHD_GLM_EOS.H"

namespace KFVM {

namespace Physics {

template <>
struct RSPhys<EquationType::MHD_GLM> {
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;
  using PrimVar = typename SimVar<EquationType::MHD_GLM>::PrimVar;
  using ConsVec = typename SimVar<EquationType::MHD_GLM>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_GLM>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};
    const int bord[] = {ConsVar::magx + ord[0], ConsVar::magx + ord[1],
                        ConsVar::magx + ord[2]};
    const Real ch = eosParams.ch_glm;

    const Real mx = U[vord[0]];
    const Real vx = V[vord[0]], vy = V[vord[1]], vz = V[vord[2]];
    const Real bx = U[bord[0]], by = U[bord[1]], bz = U[bord[2]];
    const Real et = U[ConsVar::etot], pg = V[PrimVar::prsg];
    const Real chpsi = ch * U[ConsVar::psi],
               hpsisq = 0.5 * U[ConsVar::psi] * U[ConsVar::psi];
    const Real uDb = vx * bx + vy * by + vz * bz, bsq = bx * bx + by * by + bz * bz;
    const Real pt = pg + 0.5 * bsq;

    F[ConsVar::dens] = mx;
    F[vord[0]] = mx * vx + pt - bx * bx;
    F[vord[1]] = mx * vy - bx * by;
    F[vord[2]] = mx * vz - bx * bz;
    F[ConsVar::etot] = vx * (et + pt - hpsisq) + bx * (chpsi - uDb);
    F[bord[0]] = chpsi;
    F[bord[1]] = vx * by - vy * bx;
    F[bord[2]] = vx * bz - vz * bx;
    F[ConsVar::psi] = ch * bx;
  }

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   ConsVec F) {
    PrimVec V;
    EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, U, V);
    flux(eosParams, ord, U, V, F);
  }

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
    const int bord[] = {ConsVar::magx + ord[0], ConsVar::magx + ord[1],
                        ConsVar::magx + ord[2]};
    const Real gamma = eosParams.gamma, ch = eosParams.ch_glm;

    const Real cfL = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma, uL, uL[bord[0]]);
    const Real cfR = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma, uR, uR[bord[0]]);

    // Davis style estimates with ch added in
    const Real sdL = fmin(vL[vord[0]] - fmax(ch, cfL), vR[vord[0]] - fmax(ch, cfR));
    const Real sdR = fmax(vL[vord[0]] + fmax(ch, cfL), vR[vord[0]] + fmax(ch, cfR));

    // PP estimates following Wu and Shu
    const Real rtL = sqrt(uL[ConsVar::dens]), rtR = sqrt(uR[ConsVar::dens]);
    const Real rtSum = rtL + rtR, vSum = (rtL * vL[vord[0]] + rtR * vR[vord[0]]) / rtSum;
    const Real bDiff[] = {uL[bord[0]] - uR[bord[0]], uL[bord[1]] - uR[bord[1]],
                          uL[bord[2]] - uR[bord[2]]};
    const Real bnDiff =
        sqrt(bDiff[0] * bDiff[0] + bDiff[1] * bDiff[1] + bDiff[2] * bDiff[2]) / rtSum;
    const Real aL = fmin(vL[vord[0]], vSum) - cfL - bnDiff;
    const Real aR = fmin(vR[vord[0]], vSum) + cfR + bnDiff;

    // Extremize over both
    sL = fmin(sdL, aL);
    sR = fmax(sdR, aR);

    return fmax(fabs(sL), fabs(sR));
  }

  KOKKOS_INLINE_FUNCTION
  static Real speed(const EosParameters &eosParams, const int ord[3], const ConsVec u) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;
    using Kokkos::sqrt;

    const int vlng = ConsVar::momx + ord[0];
    const int blng = ConsVar::magx + ord[0];

    const Real gamma = eosParams.gamma, ch = eosParams.ch_glm;
    const Real cf = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma, u, u[blng]);

    return fabs(u[vlng] / u[ConsVar::dens]) + fmax(cf, ch);
  }
};

template <>
struct RiemannSolver<EquationType::MHD_GLM, RSType::MHD_GLM_KEPES> {
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;
  using PrimVar = typename SimVar<EquationType::MHD_GLM>::PrimVar;
  using ConsVec = typename SimVar<EquationType::MHD_GLM>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_GLM>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static Real solve(const EosParameters &eosParams, const int dir, const ConsVec uL,
                    const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::sqrt;

    // set field orderings to collapse to 1d solver
    const int ord[] = {dir, (dir + 1) % 3, (dir + 2) % 3};
    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};
    const int bord[] = {ConsVar::magx + ord[0], ConsVar::magx + ord[1],
                        ConsVar::magx + ord[2]};
    const Real gamma = eosParams.gamma, ch = eosParams.ch_glm;

    // Wave speed estimates
    Real sL, sR, sMax;
    sMax = RSPhys<EquationType::MHD_GLM>::speeds(eosParams, ord, uL, uR, vL, vR, sL, sR);

    // Find far too many mean quantites
    const Real rhoLn = logMean(uL[ConsVar::dens], uR[ConsVar::dens]);
    const Real rhoAv = arithMean(uL[ConsVar::dens], uR[ConsVar::dens]);

    const Real uAv = arithMean(vL[vord[0]], vR[vord[0]]);
    const Real vAv = arithMean(vL[vord[1]], vR[vord[1]]);
    const Real wAv = arithMean(vL[vord[2]], vR[vord[2]]);

    const Real uuAv = arithMean(vL[vord[0]] * vL[vord[0]], vR[vord[0]] * vR[vord[0]]);
    const Real vvAv = arithMean(vL[vord[1]] * vL[vord[1]], vR[vord[1]] * vR[vord[1]]);
    const Real wwAv = arithMean(vL[vord[2]] * vL[vord[2]], vR[vord[2]] * vR[vord[2]]);

    const Real bxAv = arithMean(uL[bord[0]], uR[bord[0]]);
    const Real byAv = arithMean(uL[bord[1]], uR[bord[1]]);
    const Real bzAv = arithMean(uL[bord[2]], uR[bord[2]]);

    const Real bxxAv = arithMean(uL[bord[0]] * uL[bord[0]], uR[bord[0]] * uR[bord[0]]);
    const Real byyAv = arithMean(uL[bord[1]] * uL[bord[1]], uR[bord[1]] * uR[bord[1]]);
    const Real bzzAv = arithMean(uL[bord[2]] * uL[bord[2]], uR[bord[2]] * uR[bord[2]]);

    const Real uBxAv = arithMean(vL[vord[0]] * uL[bord[0]], vR[vord[0]] * uR[bord[0]]);
    const Real vByAv = arithMean(vL[vord[1]] * uL[bord[1]], vR[vord[1]] * uR[bord[1]]);
    const Real wBzAv = arithMean(vL[vord[2]] * uL[bord[2]], vR[vord[2]] * uR[bord[2]]);

    const Real uBxxAv = arithMean(vL[vord[0]] * uL[bord[0]] * uL[bord[0]],
                                  vR[vord[0]] * uR[bord[0]] * uR[bord[0]]);
    const Real uByyAv = arithMean(vL[vord[0]] * uL[bord[1]] * uL[bord[1]],
                                  vR[vord[0]] * uR[bord[1]] * uR[bord[1]]);
    const Real uBzzAv = arithMean(vL[vord[0]] * uL[bord[2]] * uL[bord[2]],
                                  vR[vord[0]] * uR[bord[2]] * uR[bord[2]]);

    const Real psiAv = arithMean(uL[ConsVar::psi], uR[ConsVar::psi]);
    const Real bxPsiAv =
        arithMean(uL[bord[0]] * uL[ConsVar::psi], uR[bord[0]] * uR[ConsVar::psi]);

    const Real betaL = uL[ConsVar::dens] / (Real(2.0) * vL[PrimVar::prsg]);
    const Real betaR = uR[ConsVar::dens] / (Real(2.0) * vR[PrimVar::prsg]);
    const Real betaLn = logMean(betaL, betaR);
    const Real betaAv = arithMean(betaL, betaR);

    const Real pLn = logMean(vL[PrimVar::prsg], vR[PrimVar::prsg]);
    const Real pTil = Real(0.5) * rhoAv / betaAv;
    const Real pHatTot = pTil + Real(0.5) * (bxxAv + byyAv + bzzAv);

    // Fill in flux
    F[ConsVar::dens] = rhoLn * uAv;

    F[vord[0]] = rhoLn * uAv * uAv + pHatTot - bxAv * bxAv;
    F[vord[1]] = rhoLn * uAv * vAv - bxAv * byAv;
    F[vord[2]] = rhoLn * uAv * wAv - bxAv * bzAv;

    F[bord[0]] = ch * psiAv;
    F[bord[1]] = uAv * byAv - vAv * bxAv;
    F[bord[2]] = uAv * bzAv - wAv * bxAv;

    F[ConsVar::psi] = ch * bxAv;

    F[ConsVar::etot] = (Real(1.0) / (Real(2.0) * (gamma - Real(1.0)) * betaLn) -
                        Real(0.5) * (uuAv + vvAv + wwAv)) *
                           F[ConsVar::dens] +
                       uAv * F[vord[0]] + vAv * F[vord[1]] + wAv * F[vord[2]] +
                       bxAv * F[bord[0]] + byAv * F[bord[1]] + bzAv * F[bord[2]] +
                       psiAv * F[ConsVar::psi] - Real(0.5) * (uBxxAv + uByyAv + uBzzAv) +
                       bxAv * (uBxAv + vByAv + wBzAv) - ch * bxPsiAv;

    // Add in local Lax-Friedrichs dissipation
    for (int nV = 0; nV < NUM_VARS; nV++) {
      F[nV] -= sMax * (uR[nV] - uL[nV]) / Real(2.0);
    }

    return sMax;
  }

  KOKKOS_INLINE_FUNCTION
  static Real logMean(const Real a, const Real b) {
    using Kokkos::log;

    const Real zeta = a / b, f = (zeta - Real(1.0)) / (zeta + Real(1.0)), u = f * f;
    const Real F = u < Real(0.01) ? Real(1.0) + u / Real(3.0) + u * u / Real(5.0) +
                                        u * u * u / Real(7.0)
                                  : Real(0.5) * log(zeta) / f;

    return Real(0.5) * (a + b) / F;
  }

  KOKKOS_INLINE_FUNCTION
  static Real arithMean(const Real a, const Real b) { return Real(0.5) * (a + b); }
};

} // end namespace Physics

} // end namespace KFVM

#endif
