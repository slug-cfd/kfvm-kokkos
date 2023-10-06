#include <cmath>

#include "fmt/core.h"

#include "EvalFunctionals.H"
#include "Monomials.H"

namespace KFVM {

namespace Stencil {

namespace Monomials {

double mono(EvalFunctional::Average ef, const idx_t p, const idx_t N, const double x) {
  (void)ef;
  (void)N;
  double ub = std::pow(x + 0.5, p + 1) / (p + 1);
  double lb = std::pow(x - 0.5, p + 1) / (p + 1);
  return ub - lb;
}

double mono(EvalFunctional::NDeriv ef, const idx_t p, const idx_t N, const double x) {
  (void)ef;
  if (N == 0) {
    return std::pow(x, p);
  } else if (N > p) {
    return 0.0;
  } else {
    return p * mono(ef, p - 1, x, N - 1);
  }
}

MonomialEnum::MonomialEnum(const idx_t tDeg_)
    : tDeg(tDeg_), nMono(NumMonos(tDeg_)), deg(nMono) {
  // Enumerate all monomials up to total degree tDeg
  idx_t n = 0;
#if (SPACE_DIM == 2)
  for (idx_t t = 0; t <= tDeg; t++) {
    for (idx_t i = 0; i <= t; i++) {
      deg[n][0] = i;
      deg[n][1] = t - i;
      n++;
    }
  }
#else
  for (idx_t t = 0; t <= tDeg; t++) {
    for (idx_t i = 0; i <= t; i++) {
      for (idx_t j = 0; j <= (t - i); j++) {
        deg[n][0] = i;
        deg[n][1] = j;
        deg[n][2] = t - i - j;
        n++;
      }
    }
  }
#endif
}

} // end namespace Monomials

} // end namespace Stencil

} // end namespace KFVM
