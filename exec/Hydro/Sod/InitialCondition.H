// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  (void)y;
  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real ax = Kokkos::fabs(x);

  state[SimVar::dens] = ax < 0.5 ? 1.0 : 0.125;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = (ax < 0.5 ? 1.0 : 0.1) / gamm;
}