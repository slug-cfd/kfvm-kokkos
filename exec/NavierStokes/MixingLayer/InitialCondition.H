// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::tanh;
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;

  const Real rho0 = 1.0;
  const Real u0 = (tanh(2.0 * y) + 3.0) / 8.0;
  const Real p0 = 1.0 / eosParams.gamma;

  state[ConsVar::dens] = rho0;
  state[ConsVar::momx] = rho0 * u0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p0 / gamm + 0.5 * rho0 * u0 * u0;
}
