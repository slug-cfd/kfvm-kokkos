// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;

  (void)y;
  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  const Real ax = Kokkos::fabs(x);

  state[ConsVar::dens] = ax < 0.5 ? 1.0 : 0.125;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = (ax < 0.5 ? 1.0 : 0.1) / gamm;
}
