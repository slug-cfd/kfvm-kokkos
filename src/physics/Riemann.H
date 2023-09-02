#ifndef RIEMANN_H_
#define RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"

namespace KFVM {

namespace Physics {

KOKKOS_INLINE_FUNCTION
Real hll(Real sL, Real sR, Real fL, Real fR, Real uL, Real uR) {
  return ((sR * fL - sL * fR) + sL * sR * (uR - uL)) / (sR - sL);
}

KOKKOS_INLINE_FUNCTION
Real hllState(Real sL, Real sR, Real fL, Real fR, Real uL, Real uR) {
  return ((sR * uR - sL * uL) + (fL - fR)) / (sR - sL);
}

KOKKOS_INLINE_FUNCTION
Real hllcSStar(Real rhoL, Real rhoR, Real pL, Real pR, Real uL, Real uR, Real sL,
               Real sR) {
  return ((pR - pL) + (rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR))) /
         (rhoL * (sL - uL) - rhoR * (sR - uR));
}

KOKKOS_INLINE_FUNCTION
Real hllcPStar(Real rhoL, Real rhoR, Real pL, Real pR, Real uL, Real uR, Real sL,
               Real sR) {
  const Real psiL = rhoL * (sL - uL), psiR = rhoR * (sR - uR);
  return 0.5 * (pL + pR) - 0.5 * (psiL + psiR) * (pR - pL) / (psiR - psiL) +
         psiL * psiR * (uR - uL) / (psiR - psiL);
}

KOKKOS_INLINE_FUNCTION
Real roeAvg(Real rhoL, Real rhoR, Real qL, Real qR) {
  using Kokkos::sqrt;
  return (qL * sqrt(rhoL) + qR * sqrt(rhoR)) / (sqrt(rhoL) + sqrt(rhoR));
}

} // end namespace Physics

} // end namespace KFVM

#endif
