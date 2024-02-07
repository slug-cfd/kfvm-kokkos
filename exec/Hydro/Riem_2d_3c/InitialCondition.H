// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  if (y > 0.8) {
    if (x > 0.8) {
      state[ConsVar::dens] = 1.5;
      state[ConsVar::momx] = 0.0;
      state[ConsVar::momy] = 0.0;
      state[ConsVar::momz] = 0.0;
      state[ConsVar::etot] = 1.5 / gamm;
    } else {
      state[ConsVar::dens] = 33.0 / 62.0;
      state[ConsVar::momx] = state[ConsVar::dens] * (4.0 / sqrt(11.0));
      state[ConsVar::momy] = 0.0;
      state[ConsVar::momz] = 0.0;
      state[ConsVar::etot] = 0.3 / gamm + state[ConsVar::dens] * (8.0 / 11.0);
    }
  } else {
    if (x > 0.8) {
      state[ConsVar::dens] = 33.0 / 62.0;
      state[ConsVar::momx] = 0.0;
      state[ConsVar::momy] = state[ConsVar::dens] * (4. / sqrt(11.));
      state[ConsVar::momz] = 0.0;
      state[ConsVar::etot] = 0.3 / gamm + state[ConsVar::dens] * (8.0 / 11.0);
    } else {
      state[ConsVar::dens] = 77.0 / 558.0;
      state[ConsVar::momx] = state[ConsVar::dens] * (4.0 / sqrt(11.0));
      state[ConsVar::momy] = state[ConsVar::dens] * (4.0 / sqrt(11.0));
      state[ConsVar::momz] = 0.0;
      state[ConsVar::etot] = (9.0 / 310.0) / gamm + state[ConsVar::dens] * (16.0 / 11.0);
    }
  }
}
