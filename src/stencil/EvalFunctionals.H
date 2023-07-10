#ifndef EVALFUNCTIONALS_H_
#define EVALFUNCTIONALS_H_

namespace KFVM {

namespace Stencil {

// Declare the evaluation functionals that need to be
// applied to kernel functions and polynomials
struct EvalFunctional {
  struct Point {};
  struct Average {};
  struct Deriv {};
  struct SecDeriv {};
};

} // end namespace Stencil

} // end namespace KFVM

#endif
