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

namespace MHD_8W {
using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
using PrimVar = typename SimVar<EquationType::MHD_8W>::PrimVar;
using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;
using PrimVec = typename SimVar<EquationType::MHD_8W>::PrimVec;

KOKKOS_INLINE_FUNCTION
Real mhd8wSpeed(const Real gamma, const int lng, const int bord[3], const ConsVec uL,
                const ConsVec uR, const PrimVec vL, const PrimVec vR, Real &sL,
                Real &sR) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;

  const Real cfL = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uL, uL[bord[0]]);
  const Real cfR = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uR, uR[bord[0]]);

  // Davis style estimates
  const Real sdL = fmin(vL[lng] - cfL, vR[lng] - cfR);
  const Real sdR = fmax(vL[lng] + cfL, vR[lng] + cfR);

  // PP estimates following Wu and Shu
  const Real rtL = sqrt(uL[ConsVar::dens]), rtR = sqrt(uR[ConsVar::dens]);
  const Real rtSum = rtL + rtR, vSum = (rtL * vL[lng] + rtR * vR[lng]) / rtSum;
  const Real bDiff[] = {uL[bord[0]] - uR[bord[0]], uL[bord[1]] - uR[bord[1]],
                        uL[bord[2]] - uR[bord[2]]};
  const Real bnDiff =
      sqrt(bDiff[0] * bDiff[0] + bDiff[1] * bDiff[1] + bDiff[2] * bDiff[2]) / rtSum;
  const Real aL = fmin(vL[lng], vSum) - cfL - bnDiff;
  const Real aR = fmin(vR[lng], vSum) + cfR + bnDiff;

  // Extremize over both
  sL = fmin(sdL, aL);
  sR = fmax(sdR, aR);
  const Real sMax = fmax(fabs(sL), fabs(sR));
  if constexpr (rsType == RSType::LLF) {
    sL = -sMax;
    sR = sMax;
  }

  return sMax;
}

KOKKOS_INLINE_FUNCTION
void mhd8wFlux(const Real gamma, const int vord[3], const int bord[3], const ConsVec U,
               const PrimVec V, ConsVec F) {
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
Real mhd8wHLL(const Real gamma, const int vord[3], const int bord[3], const ConsVec uL,
              const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using Kokkos::fmin;

  // Wave speed estimates
  Real sL, sR;
  const Real sMax = mhd8wSpeed(gamma, vord[0], bord, uL, uR, vL, vR, sL, sR);

  // Combine as in Batten to avoid conditionals
  const Real tR = (fmin(0.0, sR) - fmin(0.0, sL)) / (sR - sL), tL = 1.0 - tR;
  const Real tD = 0.5 * (sR * fabs(sL) - sL * fabs(sR)) / (sR - sL);

  // L/R fluxes and add together
  ConsVec FR;
  mhd8wFlux(gamma, vord, bord, uL, vL, F);
  mhd8wFlux(gamma, vord, bord, uR, vR, FR);

  for (int nV = 0; nV < NUM_VARS; nV++) {
    F[nV] = tL * F[nV] + tR * FR[nV] - tD * (uR[nV] - uL[nV]);
  }

  return sMax;
}

} // namespace MHD_8W

template <RSType rst>
struct MHD_8W_RS {};

template <>
struct MHD_8W_RS<RSType::LLF> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_8W>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    int bord[] = {ConsVar::magx, ConsVar::magy, ConsVar::magz};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    int bord[] = {ConsVar::magy, ConsVar::magz, ConsVar::magx};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    int bord[] = {ConsVar::magz, ConsVar::magx, ConsVar::magy};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }
};

template <>
struct MHD_8W_RS<RSType::HLL> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_8W>::PrimVec;

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_X(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momx, ConsVar::momy, ConsVar::momz};
    int bord[] = {ConsVar::magx, ConsVar::magy, ConsVar::magz};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Y(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momy, ConsVar::momz, ConsVar::momx};
    int bord[] = {ConsVar::magy, ConsVar::magz, ConsVar::magx};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real RiemannSolve_Z(const Real gamma, const ConsVec uL,
                                                    const ConsVec uR, const PrimVec vL,
                                                    const PrimVec vR, ConsVec F) {
    int vord[] = {ConsVar::momz, ConsVar::momx, ConsVar::momy};
    int bord[] = {ConsVar::magz, ConsVar::magx, ConsVar::magy};
    return MHD_8W::mhd8wHLL(gamma, vord, bord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
