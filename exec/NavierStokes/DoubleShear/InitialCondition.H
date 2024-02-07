// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sin;
  using Kokkos::tanh;
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real p = 1.0 / (eosParams.gamma * userParams.mach * userParams.mach);

  const Real rho = 1.0, k = userParams.k, delta = userParams.delta;
  const Real vx = y < 0.5 ? tanh(k * (y - 0.25)) : tanh(k * (0.75 - y));
  const Real vy = delta * sin(2.0 * M_PI * (x + 0.25));

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * vx;
  state[ConsVar::momy] = rho * vy;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm + 0.5 * rho * (vx * vx + vy * vy);
}
