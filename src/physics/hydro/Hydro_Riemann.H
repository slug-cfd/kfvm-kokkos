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
KOKKOS_INLINE_FUNCTION
void hydroSpeed(Real gamma, int lng, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                Real &sL, Real &sR) {
  using Hydro::SimVar;
  using Kokkos::fmax;
  using Kokkos::fmin;

  Real a = fmax(EOS<EquationType::Hydro>::soundspeed(gamma, uL),
                EOS<EquationType::Hydro>::soundspeed(gamma, uR));
  sL = fmin(uL[lng] / uL[SimVar::dens], uR[lng] / uR[SimVar::dens]) - a;
  sR = fmax(uL[lng] / uL[SimVar::dens], uR[lng] / uR[SimVar::dens]) + a;
}

KOKKOS_INLINE_FUNCTION
Real hydroHLL(Real gamma, int ord[3], const Hydro::ConsVec uL, const Hydro::ConsVec uR,
              const Hydro::PrimVec vL, const Hydro::PrimVec vR, Real F[NUM_VARS]) {
  using Hydro::ConsVec;
  using Hydro::SimVar;
  using Kokkos::fabs;
  using Kokkos::fmax;

  ConsVec FT;
  Real sL, sR;

  // Wave speed estimates
  hydroSpeed(gamma, ord[0], uL, uR, sL, sR);

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
Real hydroHLLC(Real gamma, int ord[3], const Hydro::ConsVec uL, const Hydro::ConsVec uR,
               const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
  using Hydro::ConsVec;
  using Hydro::PrimVar;
  using Hydro::SimVar;
  using Kokkos::fabs;
  using Kokkos::fmax;

  // Wave speed estimates
  Real sL, sR, sS;
  hydroSpeed(gamma, ord[0], uL, uR, sL, sR);
  sS = hllcSStar(vL[PrimVar::dens], vR[PrimVar::dens], vL[PrimVar::pres],
                 vR[PrimVar::pres], vL[ord[0]], vR[ord[0]], sL, sR);

  // Find correct zone
  if (0 < sL) {
    hydroFlux(gamma, ord, uL, vL, F);
  } else if (0 <= sS) {
    hydroFlux(gamma, ord, uL, vL, F);
    Real pre = hllcPre(uL[SimVar::dens], vL[ord[0]], sL, sS);
    Real eStar = hllcEnergy(pre, uL[SimVar::dens], uL[SimVar::etot], vL[PrimVar::pres],
                            vL[ord[0]], sL, sS);
    F[SimVar::dens] += sL * (pre - uL[SimVar::dens]);
    F[ord[0]] += sL * (pre * sS - uL[ord[0]]);
    F[ord[1]] += sL * (pre * vL[ord[1]] - uL[ord[1]]);
    F[ord[2]] += sL * (pre * vL[ord[2]] - uL[ord[2]]);
    F[SimVar::etot] += sL * (eStar - uL[SimVar::etot]);
  } else if (0 <= sR) {
    hydroFlux(gamma, ord, uR, vR, F);
    Real pre = hllcPre(uR[SimVar::dens], vR[ord[0]], sR, sS);
    Real eStar = hllcEnergy(pre, uR[SimVar::dens], uR[SimVar::etot], vR[PrimVar::pres],
                            vR[ord[0]], sR, sS);
    F[SimVar::dens] += sR * (pre - uR[SimVar::dens]);
    F[ord[0]] += sR * (pre * sS - uR[ord[0]]);
    F[ord[1]] += sR * (pre * vR[ord[1]] - uR[ord[1]]);
    F[ord[2]] += sR * (pre * vR[ord[2]] - uR[ord[2]]);
    F[SimVar::etot] += sR * (eStar - uR[SimVar::etot]);
  } else {
    hydroFlux(gamma, ord, uR, vR, F);
  }

  return fmax(fabs(sL), fabs(sR));
}

KOKKOS_INLINE_FUNCTION
Real roeAvg(Real rhoL, Real rhoR, Real qL, Real qR) {
  using Kokkos::sqrt;
  return (qL * sqrt(rhoL) + qR * sqrt(rhoR)) / (sqrt(rhoL) + sqrt(rhoR));
}

// Carbuncle suppressed Roe solver
// As given by Hu and Feng @ https://doi.org/10.1016/j.cnsns.2022.106798
KOKKOS_INLINE_FUNCTION
Real hydroROE(Real gamma, int ord[3], const Hydro::ConsVec uL, const Hydro::ConsVec uR,
              const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
  using Hydro::ConsVec;
  using Hydro::PrimVar;
  using Hydro::SimVar;
  using Kokkos::exp;
  using Kokkos::fabs;
  using Kokkos::fmax;
  using Kokkos::fmin;

  ConsVec FT;

  // Soundspeed and Roe averaged values
  Real aL = EOS<EquationType::Hydro>::soundspeed(gamma, uL);
  Real aR = EOS<EquationType::Hydro>::soundspeed(gamma, uR);
  Real rhoL = uL[SimVar::dens], rhoR = uR[SimVar::dens],
       rTil = roeAvg(rhoL, rhoR, rhoL, rhoR);
  Real aTil = roeAvg(rhoL, rhoR, aL, aR);
  Real pL = vL[PrimVar::pres], pR = vR[PrimVar::pres], pTil = roeAvg(rhoL, rhoR, pL, pR);
  Real uLngL = vL[ord[0]], uLngR = vR[ord[0]], uTil = roeAvg(rhoL, rhoR, uLngL, uLngR);
  Real vTil = roeAvg(rhoL, rhoR, vL[ord[1]], vR[ord[1]]);
  Real wTil = roeAvg(rhoL, rhoR, vL[ord[2]], vR[ord[2]]);
  Real machL = uLngL / aL, machR = uLngR / aR;
  Real mTil = fabs(roeAvg(rhoL, rhoR, machL, machR));

  // Jumps
  Real drho = rhoR - rhoL;
  Real dv = vR[ord[1]] - vL[ord[1]], dw = vR[ord[2]] - vL[ord[2]];
  Real dp = pR - pL;

  // Wave speed estimates
  Real sL = fmin(Real(0.0), uTil - aTil), sR = fmax(Real(0.0), uTil + aTil);

  // Left and right fluxes
  hydroFlux(gamma, ord, uL, vL, F);
  hydroFlux(gamma, ord, uR, vR, FT);

  // Centered HLL
  F[SimVar::dens] =
      hll(sL, sR, F[SimVar::dens], FT[SimVar::dens], uL[SimVar::dens], uR[SimVar::dens]);
  F[ord[0]] = hll(sL, sR, F[ord[0]], FT[ord[0]], uL[ord[0]], uR[ord[0]]);
  F[ord[1]] = hll(sL, sR, F[ord[1]], FT[ord[1]], uL[ord[1]], uR[ord[1]]);
  F[ord[2]] = hll(sL, sR, F[ord[2]], FT[ord[2]], uL[ord[2]], uR[ord[2]]);
  F[SimVar::etot] =
      hll(sL, sR, F[SimVar::etot], FT[SimVar::etot], uL[SimVar::etot], uR[SimVar::etot]);

  // Roe antidiffusion
  Real pdt = dp / pTil;
  Real pre = exp(-pdt * pdt) * sL * sR / ((Real(1.0) + mTil) * (sR - sL));
  Real AEW = pre * (drho - dp / (aTil * aTil));
  Real ASWv = pre * rTil * dv, ASWw = pre * rTil * dw;
  F[SimVar::dens] -= AEW;
  F[ord[0]] -= AEW * uTil;
  F[ord[1]] -= (AEW * vTil + ASWv);
  F[ord[2]] -= (AEW * wTil + ASWw);
  F[SimVar::etot] -= (AEW * (uTil * uTil + vTil * vTil + wTil * wTil) / Real(2.0) +
                      ASWv * vTil + ASWw * wTil);

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

template <>
struct HydroRS<RSType::ROE> {
  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_X(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momx, SimVar::momy, SimVar::momz};
    return hydroROE(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Y(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momy, SimVar::momz, SimVar::momx};
    return hydroROE(gamma, ord, uL, uR, vL, vR, F);
  }

  KOKKOS_INLINE_FUNCTION static Real
  RiemannSolve_Z(const Real gamma, const Hydro::ConsVec uL, const Hydro::ConsVec uR,
                 const Hydro::PrimVec vL, const Hydro::PrimVec vR, Hydro::ConsVec F) {
    using Hydro::SimVar;
    int ord[] = {SimVar::momz, SimVar::momx, SimVar::momy};
    return hydroROE(gamma, ord, uL, uR, vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
