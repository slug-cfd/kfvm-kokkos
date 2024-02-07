// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;

  const Real rho = 1.0;
  const Real u = 1.0;
  const Real p = 1.0 / (eosParams.gamma * userParams.machFree * userParams.machFree);

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm + 0.5 * rho * u * u;
}
