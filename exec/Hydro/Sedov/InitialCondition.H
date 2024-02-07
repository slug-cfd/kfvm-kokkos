// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#include "Definitions.H"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real R = userParams.R;
  const Real rsq = x * x + y * y + z * z;
  const Real p = rsq <= R * R ? gamm * gamm / (M_PI * R * R) : 1.e-5;

  // Background
  state[ConsVar::dens] = 1.0;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm;
}
