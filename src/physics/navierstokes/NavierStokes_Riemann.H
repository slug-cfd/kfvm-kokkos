#ifndef NAVIERSTOKES_RIEMANN_H_
#define NAVIERSTOKES_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "NavierStokes_EOS.H"

namespace KFVM {

namespace Physics {

KOKKOS_INLINE_FUNCTION
void navierstokesFlux(const Real gamma, int ord[3], const NavierStokes::ConsVec U,
                      const NavierStokes::PrimVec V, NavierStokes::ConsVec F) {
  using NavierStokes::PrimVar;
  using NavierStokes::SimVar;

  F[SimVar::dens] = U[ord[0]];
  F[ord[0]] = U[ord[0]] * V[ord[0]] + V[PrimVar::pres];
  F[ord[1]] = U[ord[0]] * V[ord[1]];
  F[ord[2]] = U[ord[0]] * V[ord[2]];
  F[SimVar::etot] = V[ord[0]] * (U[SimVar::etot] + V[PrimVar::pres]);
}

// Wavespeed estimates
// See Batten et. al. https://doi.org/10.1137/S1064827593260140
KOKKOS_INLINE_FUNCTION
void navierstokesSpeed(Real gamma, int lng, const NavierStokes::ConsVec uL,
                       const NavierStokes::ConsVec uR, const NavierStokes::PrimVec vL,
                       const NavierStokes::PrimVec vR, Real &sL, Real &sR) {
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;
  using NavierStokes::PrimVar;
  using NavierStokes::SimVar;

  // Roe estimates
  const Real hL = (uL[SimVar::etot] + vL[PrimVar::pres]) / uL[SimVar::dens],
             hR = (uR[SimVar::etot] + vR[PrimVar::pres]) / uR[SimVar::dens];
  const Real uTil = roeAvg(uL[SimVar::dens], uR[SimVar::dens], vL[lng], vR[lng]),
             hTil = roeAvg(uL[SimVar::dens], uR[SimVar::dens], hL, hR),
             aTil = sqrt((gamma - 1.0) * (hTil - 0.5 * uTil * uTil));

  // Batten estimate
  const Real aL = EOS<EquationType::NavierStokes>::soundspeed(gamma, uL),
             aR = EOS<EquationType::NavierStokes>::soundspeed(gamma, uR);

  sL = fmin(uL[lng] / uL[SimVar::dens] - aL, uTil - aTil);
  sR = fmax(uR[lng] / uR[SimVar::dens] + aR, uTil + aTil);
}

KOKKOS_INLINE_FUNCTION
Real navierstokesHLL(const Real gamma, int ord[3], const NavierStokes::ConsVec uL,
                     const NavierStokes::ConsVec uR, const NavierStokes::PrimVec vL,
                     const NavierStokes::PrimVec vR, Real F[NUM_VARS]) {
  using Kokkos::fabs;
  using Kokkos::fmax;
  using NavierStokes::ConsVec;
  using NavierStokes::SimVar;

  ConsVec FT;
  Real sL, sR;

  // Wave speed estimates
  navierstokesSpeed(gamma, ord[0], uL, uR, vL, vR, sL, sR);

  // Find appropriate region and fill in flux vector
  if (sR > 0) {
    navierstokesFlux(gamma, ord, uL, vL, F);
    if (sL <= 0) {
      navierstokesFlux(gamma, ord, uR, vR, FT);
      F[SimVar::dens] = hll(sL, sR, F[SimVar::dens], FT[SimVar::dens], uL[SimVar::dens],
                            uR[SimVar::dens]);
      F[ord[0]] = hll(sL, sR, F[ord[0]], FT[ord[0]], uL[ord[0]], uR[ord[0]]);
      F[ord[1]] = hll(sL, sR, F[ord[1]], FT[ord[1]], uL[ord[1]], uR[ord[1]]);
      F[ord[2]] = hll(sL, sR, F[ord[2]], FT[ord[2]], uL[ord[2]], uR[ord[2]]);
      F[SimVar::etot] = hll(sL, sR, F[SimVar::etot], FT[SimVar::etot], uL[SimVar::etot],
                            uR[SimVar::etot]);
    }
  } else {
    navierstokesFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

template <RSType rst>
struct NavierStokesRS {};

template <>
struct NavierStokesRS<RSType::HLL> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const NavierStokes::ConsVec uL,
                 const NavierStokes::ConsVec uR, const NavierStokes::PrimVec vL,
                 const NavierStokes::PrimVec vR, NavierStokes::ConsVec F) {
    using NavierStokes::SimVar;
    int ord[] = {SimVar::momx, SimVar::momy, SimVar::momz};
    return navierstokesHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const NavierStokes::ConsVec uL,
                 const NavierStokes::ConsVec uR, const NavierStokes::PrimVec vL,
                 const NavierStokes::PrimVec vR, NavierStokes::ConsVec F) {
    using NavierStokes::SimVar;
    int ord[] = {SimVar::momy, SimVar::momz, SimVar::momx};
    return navierstokesHLL(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const NavierStokes::ConsVec uL,
                 const NavierStokes::ConsVec uR, const NavierStokes::PrimVec vL,
                 const NavierStokes::PrimVec vR, NavierStokes::ConsVec F) {
    using NavierStokes::SimVar;
    int ord[] = {SimVar::momz, SimVar::momx, SimVar::momy};
    return navierstokesHLL(gamma, ord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
