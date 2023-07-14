// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::exp;
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real rho = 1.0 + exp(-50.0 * (x * x + (y - 1.0) * (y - 1.0)));
  const Real vx = -y / sqrt(x * x + y * y), vy = x / sqrt(x * x + y * y);

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * vx;
  state[SimVar::momy] = rho * vy;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = 1.0 / (eosParams.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy);
}
