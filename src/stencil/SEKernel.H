#ifndef SEKERNEL_H_
#define SEKERNEL_H_

#include "EvalFunctionals.H"
#include "KernelTypes.H"

namespace KFVM {

namespace Stencil {

namespace SE {

// One dimensional kernel functions
static double K(EvalFunctional::Average ef, const double eps, const idx_t N,
                const double x) {
  (void)ef;
  (void)N;
  auto pf = std::sqrt(M_PI) / (2.0 * eps);
  return pf * (std::erf(eps * (x + 0.5)) - std::erf(eps * (x - 0.5)));
}

static double K(EvalFunctional::NDeriv ef, const double eps, const idx_t N,
                const double x) {
  (void)ef;
  const double sgn = N % 2 == 0 ? 1.0 : -1.0,
               Hn = sgn * std::pow(eps, N) * std::hermite(N, eps * x);
  return Hn * std::exp(-eps * eps * x * x);
}

} // namespace SE

template <>
struct Kernel<KernelType::SE> {
  // Define higher dimensional kernel functions through 1D calls
  template <class efX, class efY>
  static double K(const double eps, const std::array<idx_t, 2> &N, const double x,
                  const double y) {
    return SE::K(efX(), eps, N[0], x) * SE::K(efY(), eps, N[1], y);
  }

  template <class efX, class efY, class efZ>
  static double K(const double eps, const std::array<idx_t, 3> &N, const double x,
                  const double y, const double z) {
    return SE::K(efX(), eps, N[0], x) * SE::K(efY(), eps, N[1], y) *
           SE::K(efZ(), eps, N[2], z);
  }
};

} // end namespace Stencil

} // end namespace KFVM

#endif
