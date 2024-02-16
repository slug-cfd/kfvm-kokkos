#ifndef MHD_8W_RIEMANN_H_
#define MHD_8W_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "MHD_8W_EOS.H"

namespace KFVM {

namespace Physics {

template <>
struct RSPhys<EquationType::MHD_8W> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using PrimVar = typename SimVar<EquationType::MHD_8W>::PrimVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_8W>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    (void)eosParams;

    const int vord[] = {ConsVar::momx + ord[0], ConsVar::momx + ord[1],
                        ConsVar::momx + ord[2]};
    const int bord[] = {ConsVar::magx + ord[0], ConsVar::magx + ord[1],
                        ConsVar::magx + ord[2]};

    const Real mx = U[vord[0]];
    const Real vx = V[vord[0]], vy = V[vord[1]], vz = V[vord[2]];
    const Real bx = U[bord[0]], by = U[bord[1]], bz = U[bord[2]];
    const Real et = U[ConsVar::etot], pg = V[PrimVar::prsg];
    const Real uDb = vx * bx + vy * by + vz * bz, bsq = bx * bx + by * by + bz * bz;
    const Real pt = pg + 0.5 * bsq;

    F[ConsVar::dens] = mx;
    F[vord[0]] = mx * vx + pt - bx * bx;
    F[vord[1]] = mx * vy - bx * by;
    F[vord[2]] = mx * vz - bx * bz;
    F[ConsVar::etot] = vx * (et + pt) - bx * uDb;
    F[bord[0]] = 0.0;
    F[bord[1]] = vx * by - vy * bx;
    F[bord[2]] = vx * bz - vz * bx;
  }

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   ConsVec F) {
    PrimVec V;
    EOS<EquationType::MHD_8W>::cons2prim(eosParams.gamma, U, V);
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
    const Real gamma = eosParams.gamma;

    const Real cfL = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uL, uL[bord[0]]);
    const Real cfR = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uR, uR[bord[0]]);

    // Davis style estimates with ch added in
    const Real sdL = fmin(vL[vord[0]] - cfL, vR[vord[0]] - cfR);
    const Real sdR = fmax(vL[vord[0]] + cfL, vR[vord[0]] + cfR);

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

    const Real gamma = eosParams.gamma;
    const Real cf = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, u, u[blng]);

    return fabs(u[vlng] / u[ConsVar::dens]) + cf;
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
