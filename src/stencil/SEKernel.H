#ifndef SEKERNEL_H_
#define SEKERNEL_H_

#include "EvalFunctionals.H"
#include "KernelTypes.H"

namespace KFVM {

namespace Stencil {

namespace SE {

// One dimensional kernel functions
static double K(EvalFunctional::Point ef, double eps, const double dx) {
  (void)ef;
  return std::exp(-eps * eps * dx * dx);
}

static double K(EvalFunctional::Average ef, double eps, const double dx) {
  (void)ef;
  auto pf = std::sqrt(M_PI) / (2.0 * eps);
  return pf * (std::erf(eps * (dx + 0.5)) - std::erf(eps * (dx - 0.5)));
}

static double K(EvalFunctional::Deriv ef, double eps, const double dx) {
  (void)ef;
  auto epep = eps * eps;
  return -2.0 * epep * dx * std::exp(-epep * dx * dx);
}

static double K(EvalFunctional::SecDeriv ef, double eps, const double dx) {
  (void)ef;
  auto epep = eps * eps;
  return 2.0 * epep * (2.0 * epep * dx * dx - 1.0) * std::exp(-epep * dx * dx);
}

} // namespace SE

template <>
struct Kernel<KernelType::SE> {
  // Define higher dimensional kernel functions through 1D calls
  template <class efX, class efY>
  static double K(double eps, const double dx, const double dy) {
    return SE::K(efX(), eps, dx) * SE::K(efY(), eps, dy);
  }

  template <class efX, class efY, class efZ>
  static double K(double eps, const double dx, const double dy, const double dz) {
    return SE::K(efX(), eps, dx) * SE::K(efY(), eps, dy) * SE::K(efZ(), eps, dz);
  }
};

} // end namespace Stencil

} // end namespace KFVM

#endif
