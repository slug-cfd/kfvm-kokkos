#ifndef SEKERNEL_H_
#define SEKERNEL_H_

#include "EvalFunctionals.H"

namespace KFVM {

namespace Stencil {

namespace SE {

// Declare one dimensional kernel functions
double K(EvalFunctional::Point, double, const double);
double K(EvalFunctional::Average, double, const double);
double K(EvalFunctional::Deriv, double, const double);
double K(EvalFunctional::SecDeriv, double, const double);

// Define higher dimensional kernel functions through 1D calls
template <class efX, class efY>
double K(double eps, const double dx, const double dy) {
  return K(efX(), eps, dx) * K(efY(), eps, dy);
}

template <class efX, class efY, class efZ>
double K(double eps, const double dx, const double dy, const double dz) {
  return K(efX(), eps, dx) * K(efY(), eps, dy) * K(efZ(), eps, dz);
}

} // end namespace SE

} // end namespace Stencil

} // end namespace KFVM

#endif
