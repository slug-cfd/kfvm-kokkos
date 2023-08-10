// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real dr = userParams.densityRatio;
  const Real gam = eosParams.gamma, gamm = gam - 1.0;

  state[SimVar::dens] = gam / dr;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = 1.0 / gamm;
}
