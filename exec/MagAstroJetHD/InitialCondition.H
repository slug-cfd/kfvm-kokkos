// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using Physics::MHD_GLM::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real gam = eosParams.gamma, gamm = gam - 1.0;
  const Real dr = userParams.densityRatio, by = sqrt(2.0 / userParams.beta);

  state[SimVar::dens] = gam / dr;

  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;

  state[SimVar::magx] = 0.0;
  state[SimVar::magy] = by;
  state[SimVar::magz] = 0.0;

  state[SimVar::etot] = 1.0 / gamm + 0.5 * by * by;

  state[SimVar::psi] = 0.0;
}
