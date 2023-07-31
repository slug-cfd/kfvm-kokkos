#ifndef HYDRO_RIEMANN_H_
#define HYDRO_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Geometry.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "Hydro_EOS.H"

namespace KFVM {

namespace Physics {

KOKKOS_INLINE_FUNCTION
void hydroFlux(Real gamma, const Vec3 &metric, const Real J, const Real vNorm,
               const Hydro::ConsVec U, const Hydro::PrimVec V, Hydro::ConsVec F) {
  using Hydro::PrimVar;
  using Hydro::SimVar;

  F[SimVar::dens] = U[SimVar::dens] * vNorm;
  F[SimVar::momx] = U[SimVar::momx] * vNorm + J * metric.v1 * V[PrimVar::pres];
  F[SimVar::momy] = U[SimVar::momy] * vNorm + J * metric.v2 * V[PrimVar::pres];
  F[SimVar::momz] = U[SimVar::momz] * vNorm + J * metric.v3 * V[PrimVar::pres];
  F[SimVar::etot] = (U[SimVar::etot] + J * V[PrimVar::pres]) * vNorm;
}

// Wavespeed estimates
KOKKOS_INLINE_FUNCTION
void hydroSpeed(const Real gamma, const Vec3 &metric, const Real J, const Real vnL,
                const Real vnR, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                Real &sL, Real &sR) {
  using Hydro::SimVar;
  using Kokkos::fmax;
  using Kokkos::fmin;
  using Kokkos::sqrt;

  Real a = fmax(EOS<EquationType::Hydro>::soundspeed(gamma, J, uL),
                EOS<EquationType::Hydro>::soundspeed(gamma, J, uR));

  Real M = sqrt(metric.v1 * metric.v1 + metric.v2 * metric.v2 + metric.v3 * metric.v3);

  sL = fmin(vnL, vnR) - a * M;
  sR = fmax(vnL, vnR) + a * M;
}

KOKKOS_INLINE_FUNCTION
Real hydroHLL(const Real gamma, const Vec3 &metric, const Real J, const Hydro::ConsVec uL,
              const Hydro::ConsVec uR, const Hydro::PrimVec vL, const Hydro::PrimVec vR,
              Real F[NUM_VARS]) {
  using Hydro::ConsVec;
  using Hydro::PrimVar;
  using Kokkos::fabs;
  using Kokkos::fmax;

  ConsVec FT;
  Real sL, sR;

  // Normal velocity in physical space
  Real vnL = metric.v1 * vL[PrimVar::velx] + metric.v2 * vL[PrimVar::vely] +
             metric.v3 * vL[PrimVar::velz];
  Real vnR = metric.v1 * vR[PrimVar::velx] + metric.v2 * vR[PrimVar::vely] +
             metric.v3 * vR[PrimVar::velz];

  // Wave speed estimates
  hydroSpeed(gamma, metric, J, vnL, vnR, uL, uR, sL, sR);

  // Find appropriate region and fill in flux vector
  if (sR > 0) {
    hydroFlux(gamma, metric, J, vnL, uL, vL, F);
    if (sL <= 0) {
      hydroFlux(gamma, metric, J, vnR, uR, vR, FT);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        F[nV] = hll(sL, sR, F[nV], FT[nV], uL[nV], uR[nV]);
      }
    }
  } else {
    hydroFlux(gamma, metric, J, vnR, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

KOKKOS_INLINE_FUNCTION
Real hllcSStar(Real rhoL, Real rhoR, Real pL, Real pR, Real uL, Real uR, Real sL,
               Real sR) {
  return ((pR - pL) + (rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR))) /
         (rhoL * (sL - uL) - rhoR * (sR - uR));
}

KOKKOS_INLINE_FUNCTION
Real hllcPre(Real rho, Real u, Real s, Real sStar) { return rho * (s - u) / (s - sStar); }

KOKKOS_INLINE_FUNCTION
Real hllcEnergy(Real pre, Real rhoK, Real EK, Real pK, Real uK, Real sK, Real sS) {
  return pre * (EK / rhoK + (sS - uK) * (sS + pK / (rhoK * (sK - uK))));
}

KOKKOS_INLINE_FUNCTION
Real hydroHLLC(Real gamma, int ord[3], const Vec3 &metric, const Real J,
               const Hydro::ConsVec uL, const Hydro::ConsVec uR, const Hydro::PrimVec vL,
               const Hydro::PrimVec vR, Hydro::ConsVec F) {
  using Hydro::ConsVec;
  using Hydro::PrimVar;
  using Hydro::SimVar;
  using Kokkos::fabs;
  using Kokkos::fmax;

  // Normal velocity in physical space
  Real vnL = metric.v1 * vL[PrimVar::velx] + metric.v2 * vL[PrimVar::vely] +
             metric.v3 * vL[PrimVar::velz];
  Real vnR = metric.v1 * vR[PrimVar::velx] + metric.v2 * vR[PrimVar::vely] +
             metric.v3 * vR[PrimVar::velz];

  // Wave speed estimates
  Real sL, sR, sS;
  hydroSpeed(gamma, metric, J, vnL, vnR, uL, uR, sL, sR);
  sS = hllcSStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                 vR[PrimVar::pres], vL[ord[0]], vR[ord[0]], sL, sR);

  // Find correct zone
  if (0 < sL) {
    hydroFlux(gamma, metric, J, vnL, uL, vL, F);
  } else if (0 <= sS) {
    hydroFlux(gamma, metric, J, vnL, uL, vL, F);
    Real pre = hllcPre(uL[SimVar::dens], vL[ord[0]], sL, sS);
    Real eStar = hllcEnergy(pre, uL[SimVar::dens], uL[SimVar::etot], vL[PrimVar::pres],
                            vL[ord[0]], sL, sS);
    F[SimVar::dens] += sL * (pre - uL[SimVar::dens]);
    F[ord[0]] += sL * (pre * sS - uL[ord[0]]);
    F[ord[1]] += sL * (pre * vL[ord[1]] - uL[ord[1]]);
    F[ord[2]] += sL * (pre * vL[ord[2]] - uL[ord[2]]);
    F[SimVar::etot] += sL * (eStar - uL[SimVar::etot]);
  } else if (0 <= sR) {
    hydroFlux(gamma, metric, J, vnR, uR, vR, F);
    Real pre = hllcPre(uR[SimVar::dens], vR[ord[0]], sR, sS);
    Real eStar = hllcEnergy(pre, uR[SimVar::dens], uR[SimVar::etot], vR[PrimVar::pres],
                            vR[ord[0]], sR, sS);
    F[SimVar::dens] += sR * (pre - uR[SimVar::dens]);
    F[ord[0]] += sR * (pre * sS - uR[ord[0]]);
    F[ord[1]] += sR * (pre * vR[ord[1]] - uR[ord[1]]);
    F[ord[2]] += sR * (pre * vR[ord[2]] - uR[ord[2]]);
    F[SimVar::etot] += sR * (eStar - uR[SimVar::etot]);
  } else {
    hydroFlux(gamma, metric, J, vnR, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

template <RSType rst>
struct HydroRS {};

template <>
struct HydroRS<RSType::HLL> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::PrimVar;
    return hydroHLL(gamma, metric, J, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::PrimVar;
    return hydroHLL(gamma, metric, J, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::PrimVar;
    return hydroHLL(gamma, metric, J, uL, uR, vL, vR, F);
  }
};

template <>
struct HydroRS<RSType::HLLC> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momx, SimVar::momy, SimVar::momz};
    return hydroHLLC(gamma, ord, metric, J, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momy, SimVar::momz, SimVar::momx};
    return hydroHLLC(gamma, ord, metric, J, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const Vec3 &metric, const Real J,
                 const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momz, SimVar::momx, SimVar::momy};
    return hydroHLLC(gamma, ord, metric, J, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
