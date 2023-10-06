#ifndef PHSKERNEL_H_
#define PHSKERNEL_H_

#include <cmath>

#include "../numeric/Numeric.H"
#include "EvalFunctionals.H"
#include "KernelTypes.H"

namespace KFVM {

namespace Stencil {

namespace PHS {

// Templates needing specialization for all evaluation functionals
template <class efX, class efY, class efZ>
static double K(const double, const double, const double, const double) {
  return NAN;
}

// Pointwise evaluations
template <>
double K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  const double dsq = dx * dx + dy * dy + dz * dz;
  return std::pow(dsq, N / 2.0);
}

// Average evaluation
template <>
double K<EvalFunctional::Average, EvalFunctional::Average, EvalFunctional::Average>(
    const double N, const double dx, const double dy, const double dz) {
  double itg = 0.0;
  for (int nX = 0; nX < 5; nX++) {
    const double abX = Numeric::QuadRuleLUT<5>::ab[nX] + dx;
    const double wtX = Numeric::QuadRuleLUT<5>::wt[nX];
    for (int nY = 0; nY < 5; nY++) {
      const double abY = Numeric::QuadRuleLUT<5>::ab[nY] + dy;
      const double wtY = Numeric::QuadRuleLUT<5>::wt[nY];
      for (int nZ = 0; nZ < 5; nZ++) {
        const double abZ = Numeric::QuadRuleLUT<5>::ab[nZ] + dz;
        const double wtZ = Numeric::QuadRuleLUT<5>::wt[nZ];
        itg += wtX * wtY * wtZ *
               K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(
                   N, abX, abY, abZ);
      }
    }
  }
  return itg;
}

// First derivatives
template <>
double K<EvalFunctional::Deriv, EvalFunctional::Point, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * dx *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 2, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::Deriv, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * dy *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 2, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Deriv>(
    const double N, const double dx, const double dy, const double dz) {
  return N * dz *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 2, dx,
                                                                                dy, dz);
}

// Second derivatives
template <>
double K<EvalFunctional::SecondDeriv, EvalFunctional::Point, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * ((N - 1.0) * dx * dx + dy * dy + dz * dz) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 4, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::SecondDeriv, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * (dx * dx + (N - 1.0) * dy * dy + dz * dz) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 4, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::SecondDeriv>(
    const double N, const double dx, const double dy, const double dz) {
  return N * (dx * dx + dy * dy + (N - 1.0) * dz * dz) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 4, dx,
                                                                                dy, dz);
}

// Third derivatives
template <>
double K<EvalFunctional::ThirdDeriv, EvalFunctional::Point, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * (N - 2.0) * dx * ((N - 1.0) * dx * dx + 3.0 * dy * dy + 3.0 * dz * dz) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 6, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::ThirdDeriv, EvalFunctional::Point>(
    const double N, const double dx, const double dy, const double dz) {
  return N * (N - 2.0) * dy * ((N - 1.0) * dy * dy + 3.0 * dx * dx + 3.0 * dz * dz) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 6, dx,
                                                                                dy, dz);
}

template <>
double K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::ThirdDeriv>(
    const double N, const double dx, const double dy, const double dz) {
  return N * (N - 2.0) * dz * ((N - 1.0) * dz * dz + 3.0 * dy * dy + 3.0 * dx * dx) *
         K<EvalFunctional::Point, EvalFunctional::Point, EvalFunctional::Point>(N - 6, dx,
                                                                                dy, dz);
}

// 2D mostly inherits from 3d
template <class efX, class efY>
static double K(const double N, const double dx, const double dy) {
  return K<efX, efY, EvalFunctional::Point>(N, dx, dy, 0.0);
}

// Though averaging is special
template <>
double K<EvalFunctional::Average, EvalFunctional::Average>(const double N,
                                                           const double dx,
                                                           const double dy) {
  double itg = 0.0;
  for (int nX = 0; nX < 5; nX++) {
    const double abX = Numeric::QuadRuleLUT<5>::ab[nX] + dx;
    const double wtX = Numeric::QuadRuleLUT<5>::wt[nX];
    for (int nY = 0; nY < 5; nY++) {
      const double abY = Numeric::QuadRuleLUT<5>::ab[nY] + dy;
      const double wtY = Numeric::QuadRuleLUT<5>::wt[nY];
      itg += wtX * wtY * K<EvalFunctional::Point, EvalFunctional::Point>(N, abX, abY);
    }
  }
  return itg;
}

} // namespace PHS

template <>
struct Kernel<KernelType::PHS> {
  template <class efX, class efY>
  static double K(const double N, const double dx, const double dy) {
    return PHS::K<efX, efY>(N, dx, dy);
  }

  template <class efX, class efY, class efZ>
  static double K(const double N, const double dx, const double dy, const double dz) {
    return PHS::K<efX, efY, efZ>(N, dx, dy, dz);
  }
};

} // end namespace Stencil

} // end namespace KFVM

#endif
