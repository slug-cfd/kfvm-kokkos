// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::sin;
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rho = 1.0, V = 1.0;
  const Real theta = atan2(y, x), vx = -V * cos(theta), vy = -V * sin(theta);
  const Real P = 1.e-6;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * vx;
  state[ConsVar::momy] = rho * vy;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = P / gamm + 0.5 * rho * (vx * vx + vy * vy);
}
