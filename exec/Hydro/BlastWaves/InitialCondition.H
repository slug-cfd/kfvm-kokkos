// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;

  const Real gamm = eosParams.gamma - 1.0;
  const Real p = x <= 0.1 ? 1000.0 : (x >= 0.9 ? 100.0 : 0.01);

  state[ConsVar::dens] = 1.0;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm;
}
