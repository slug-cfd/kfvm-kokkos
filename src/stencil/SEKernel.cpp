#include <cmath>

#include "EvalFunctionals.H"
#include "SEKernel.H"

namespace KFVM {

namespace Stencil {

namespace SE {
// One dimensional kernel functions
double K(EvalFunctional::Point ef, double eps, const double dx) {
  (void)ef;
  return std::exp(-eps * eps * dx * dx);
}

double K(EvalFunctional::Average ef, double eps, const double dx) {
  (void)ef;
  auto pf = std::sqrt(M_PI) / (2.0 * eps);
  return pf * (std::erf(eps * (dx + 0.5)) - std::erf(eps * (dx - 0.5)));
}

double K(EvalFunctional::Deriv ef, double eps, const double dx) {
  (void)ef;
  auto epep = eps * eps;
  return -2.0 * epep * dx * std::exp(-epep * dx * dx);
}

double K(EvalFunctional::SecDeriv ef, double eps, const double dx) {
  (void)ef;
  auto epep = eps * eps;
  return 2.0 * epep * (2.0 * epep * dx * dx - 1.0) * std::exp(-epep * dx * dx);
}

} // end namespace SE

} // end namespace Stencil

} // end namespace KFVM
