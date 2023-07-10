// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sin;
  using Physics::Hydro::SimVar;

  (void)y;
  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  const Real rho = x < 0.5 ? 3.857143 : 1.0 + 0.2 * sin(5.0 * (x - 4.5));
  const Real u = x < 0.5 ? 2.629369 : 0.0;
  const Real p = x < 0.5 ? 10.33333 : 1.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / gamm + 0.5 * rho * u * u;
}
