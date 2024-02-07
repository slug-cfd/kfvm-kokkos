// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sin;
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)y;
  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  const Real rho = x < 0.5 ? 3.857143 : 1.0 + 0.2 * sin(5.0 * (x - 4.5));
  const Real u = x < 0.5 ? 2.629369 : 0.0;
  const Real p = x < 0.5 ? 10.33333 : 1.0;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm + 0.5 * rho * u * u;
}
