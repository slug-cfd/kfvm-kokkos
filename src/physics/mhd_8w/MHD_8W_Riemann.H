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

KOKKOS_INLINE_FUNCTION
Real mhd8wSpeed(const Real gamma, int lng, int bord[3], const MHD_8W::ConsVec uL,
                const MHD_8W::ConsVec uR, const MHD_8W::PrimVec vL,
                const MHD_8W::PrimVec vR, Real &sL, Real &sR) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;
  using MHD_8W::SimVar;

  const Real cfL = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uL, uL[bord[0]]);
  const Real cfR = EOS<EquationType::MHD_8W>::fastMagSonic(gamma, uR, uR[bord[0]]);

  // PP estimates following Wu and Shu
  const Real rtL = sqrt(uL[SimVar::dens]), rtR = sqrt(uR[SimVar::dens]);
  const Real rtSum = rtL + rtR, vSum = (rtL * vL[lng] + rtR * vR[lng]) / rtSum;
  const Real bDiff[] = {uL[bord[0]] - uR[bord[0]], uL[bord[1]] - uR[bord[1]],
                        uL[bord[2]] - uR[bord[2]]};
  const Real bnDiff =
      sqrt(bDiff[0] * bDiff[0] + bDiff[1] * bDiff[1] + bDiff[2] * bDiff[2]) / rtSum;
  sL = fmin(vL[lng], vSum) - cfL - bnDiff;
  sR = fmin(vR[lng], vSum) + cfR + bnDiff;

  return fmax(fabs(sL), fabs(sR));
}

KOKKOS_INLINE_FUNCTION
void mhd8wFlux(const Real gamma, int vord[3], int bord[3], const MHD_8W::ConsVec U,
               const MHD_8W::PrimVec V, MHD_8W::ConsVec F) {
  using MHD_8W::PrimVar;
  using MHD_8W::SimVar;

  const Real mx = U[vord[0]];
  const Real vx = V[vord[0]], vy = V[vord[1]], vz = V[vord[2]];
  const Real bx = U[bord[0]], by = U[bord[1]], bz = U[bord[2]];
  const Real et = U[SimVar::etot], pg = V[PrimVar::prsg];
  const Real uDb = vx * bx + vy * by + vz * bz, bsq = bx * bx + by * by + bz * bz;
  const Real pt = pg + 0.5 * bsq;

  F[SimVar::dens] = mx;
  F[vord[0]] = mx * vx + pt - bx * bx;
  F[vord[1]] = mx * vy - bx * by;
  F[vord[2]] = mx * vz - bx * bz;
  F[SimVar::etot] = vx * (et + pt) - bx * uDb;
  F[bord[0]] = 0.0;
  F[bord[1]] = vx * by - vy * bx;
  F[bord[2]] = vx * bz - vz * bx;
}

KOKKOS_INLINE_FUNCTION
Real mhd8wLLF(const Real gamma, int vord[3], int bord[3], const MHD_8W::ConsVec uL,
              const MHD_8W::ConsVec uR, const MHD_8W::PrimVec vL,
              const MHD_8W::PrimVec vR, MHD_8W::ConsVec F) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using MHD_8W::ConsVec;

  // Wave speed estimates
  Real sL, sR;
  const Real sMax = mhd8wSpeed(gamma, vord[0], bord, uL, uR, vL, vR, sL, sR);

  // Left and right fluxes
  ConsVec FR;
  mhd8wFlux(gamma, vord, bord, uL, vL, F);
  mhd8wFlux(gamma, vord, bord, uR, vR, FR);

  // Add in local Lax-Friedrichs dissipation
  for (int nV = 0; nV < NUM_VARS; nV++) {
    F[nV] = 0.5 * ((F[nV] + FR[nV]) - sMax * (uR[nV] - uL[nV]));
  }

  return sMax;
}

template <RSType rst>
struct MHD_8W_RS {};

template <>
struct MHD_8W_RS<RSType::LLF> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const MHD_8W::ConsVec uL, const MHD_8W::ConsVec uR,
                 const MHD_8W::PrimVec vL, const MHD_8W::PrimVec vR, MHD_8W::ConsVec F) {
    int vord[] = {MHD_8W::SimVar::momx, MHD_8W::SimVar::momy, MHD_8W::SimVar::momz};
    int bord[] = {MHD_8W::SimVar::magx, MHD_8W::SimVar::magy, MHD_8W::SimVar::magz};
    return mhd8wLLF(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const MHD_8W::ConsVec uL, const MHD_8W::ConsVec uR,
                 const MHD_8W::PrimVec vL, const MHD_8W::PrimVec vR, MHD_8W::ConsVec F) {
    int vord[] = {MHD_8W::SimVar::momy, MHD_8W::SimVar::momz, MHD_8W::SimVar::momx};
    int bord[] = {MHD_8W::SimVar::magy, MHD_8W::SimVar::magz, MHD_8W::SimVar::magx};
    return mhd8wLLF(gamma, vord, bord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const MHD_8W::ConsVec uL, const MHD_8W::ConsVec uR,
                 const MHD_8W::PrimVec vL, const MHD_8W::PrimVec vR, MHD_8W::ConsVec F) {
    int vord[] = {MHD_8W::SimVar::momz, MHD_8W::SimVar::momx, MHD_8W::SimVar::momy};
    int bord[] = {MHD_8W::SimVar::magz, MHD_8W::SimVar::magx, MHD_8W::SimVar::magy};
    return mhd8wLLF(gamma, vord, bord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif