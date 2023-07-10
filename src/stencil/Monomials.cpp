#include <cmath>

#include "EvalFunctionals.H"

namespace KFVM {

namespace Stencil {

namespace Monomials {

double mono(EvalFunctional::Point ef, int p, const double x) {
  (void)ef;
  return std::pow(x, p);
}

double mono(EvalFunctional::Average ef, int p, const double x) {
  (void)ef;
  double ub = std::pow(x + 0.5, p + 1) / (p + 1);
  double lb = std::pow(x - 0.5, p + 1) / (p + 1);
  return ub - lb;
}

double mono(EvalFunctional::Deriv ef, int p, const double x) {
  (void)ef;
  if (p > 0) {
    return p * std::pow(x, p - 1);
  } else {
    return 0.0;
  }
}

double mono(EvalFunctional::SecDeriv ef, int p, const double x) {
  (void)ef;
  if (p > 1) {
    return p * (p - 1) * std::pow(x, p - 2);
  } else {
    return 0.0;
  }
}

} // end namespace Monomials

} // end namespace Stencil

} // end namespace KFVM
