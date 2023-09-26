#ifndef RIEMANN_H_
#define RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"

namespace KFVM {

namespace Physics {

KOKKOS_INLINE_FUNCTION
Real hll(const Real sL, const Real sR, const Real fL, const Real fR, const Real uL,
         const Real uR) {
  return ((sR * fL - sL * fR) + sL * sR * (uR - uL)) / (sR - sL);
}

KOKKOS_INLINE_FUNCTION
Real hllState(const Real sL, const Real sR, const Real fL, const Real fR, const Real uL,
              const Real uR) {
  return ((sR * uR - sL * uL) + (fL - fR)) / (sR - sL);
}

KOKKOS_INLINE_FUNCTION
Real hllcSStar(const Real rhoL, const Real rhoR, const Real pL, const Real pR,
               const Real uL, const Real uR, const Real sL, const Real sR) {
  return ((pR - pL) + (rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR))) /
         (rhoL * (sL - uL) - rhoR * (sR - uR));
}

KOKKOS_INLINE_FUNCTION
Real hllcPStar(const Real rhoL, const Real rhoR, const Real pL, const Real pR,
               const Real uL, const Real uR, const Real sL, const Real sR,
               const Real sS) {
  const Real psiL = rhoL * (sL - uL), psiR = rhoR * (sR - uR);
  return 0.5 * ((pL + pR) + psiL * (sS - uL) + psiR * (sS - uR));
}

KOKKOS_INLINE_FUNCTION
Real roeAvg(const Real rhoL, const Real rhoR, const Real qL, const Real qR) {
  using Kokkos::sqrt;
  return (qL * sqrt(rhoL) + qR * sqrt(rhoR)) / (sqrt(rhoL) + sqrt(rhoR));
}

} // end namespace Physics

} // end namespace KFVM

#endif
