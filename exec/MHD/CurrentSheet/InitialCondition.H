// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::fabs;
  using Kokkos::sin;
  using Physics::MHD_GLM::SimVar;

  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;

  const Real rho = 1.0, p = 0.5 * userParams.beta;
  const Real u = userParams.A * sin(2.0 * M_PI * y);
  const Real by = fabs(x) > 0.25 ? 1.0 : -1.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / (eosParams.gamma - 1.0) + 0.5 * rho * u * u + 0.5 * by * by;
  state[SimVar::magx] = 0.0;
  state[SimVar::magy] = by;
  state[SimVar::magz] = 0.0;
  state[SimVar::psi] = 0.0;
}
