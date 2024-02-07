// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::sin;
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::LinAdv>::ConsVar;

  (void)z;

  const Real xpar = x * cos(userParams.ang) + y * sin(userParams.ang);

  state[ConsVar::u] = sin(2.0 * M_PI * userParams.freq * xpar);
}
