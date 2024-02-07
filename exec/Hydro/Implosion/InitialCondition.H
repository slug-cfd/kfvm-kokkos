// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::fabs;
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;

  if ((fabs(x) + fabs(y) + fabs(z)) <= 0.15) {
    state[ConsVar::dens] = 0.125;
    state[ConsVar::etot] = 0.14 / gamm;
  } else {
    state[ConsVar::dens] = 1.0;
    state[ConsVar::etot] = 1.0 / gamm;
  }
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
}
