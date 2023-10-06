#ifndef MONONOMIALS_H_
#define MONONOMIALS_H_

#include <array>

#include "../Types.H"
#include "EvalFunctionals.H"

namespace KFVM {

namespace Stencil {

namespace Monomials {

// Declare univariate monomials under each evaluation functional
double mono(EvalFunctional::Average, const idx_t, const idx_t, const double);
double mono(EvalFunctional::NDeriv, const idx_t, const idx_t, const double);

// Define multivariate monomials through 1d calls
template <class efX, class efY>
double mono(const std::array<idx_t, 2> &p, const std::array<idx_t, 2> &N, const double x,
            const double y) {
  return mono(efX(), p[0], N[0], x) * mono(efY(), p[1], N[1], y);
}

template <class efX, class efY, class efZ>
double mono(const std::array<idx_t, 3> &p, const std::array<idx_t, 3> &N, const double x,
            const double y, const double z) {
  return mono(efX(), p[0], N[0], x) * mono(efY(), p[1], N[1], y) *
         mono(efZ(), p[2], N[2], z);
}

constexpr idx_t NumMonos(const idx_t t) {
#if (SPACE_DIM == 2)
  return (t + 1) * (t + 2) / 2;
#else
  return (t + 1) * (t + 2) * (t + 3) / 6;
#endif
}

// Enumeration of used monomials up to given total degree
struct MonomialEnum {
  const idx_t tDeg, nMono;
  std::vector<std::array<idx_t, SPACE_DIM>> deg;

  MonomialEnum(const idx_t);
};

} // end namespace Monomials

} // end namespace Stencil

} // end namespace KFVM

#endif
