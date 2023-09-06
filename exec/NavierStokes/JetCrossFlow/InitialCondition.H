// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::NavierStokes::SimVar;

  const Real gamm = eosParams.gamma - 1.0;

  const Real rho = 1.0;
  const Real u = 1.0;
  const Real p = 1.0 / (eosParams.gamma * userParams.machFree * userParams.machFree);

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / gamm + 0.5 * rho * u * u;
}
