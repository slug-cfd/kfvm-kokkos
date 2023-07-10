// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  if (y > 0.8) {
    if (x > 0.8) {
      state[SimVar::dens] = 1.5;
      state[SimVar::momx] = 0.0;
      state[SimVar::momy] = 0.0;
      state[SimVar::momz] = 0.0;
      state[SimVar::etot] = 1.5 / gamm;
    } else {
      state[SimVar::dens] = 33.0 / 62.0;
      state[SimVar::momx] = state[SimVar::dens] * (4.0 / sqrt(11.0));
      state[SimVar::momy] = 0.0;
      state[SimVar::momz] = 0.0;
      state[SimVar::etot] = 0.3 / gamm + state[SimVar::dens] * (8.0 / 11.0);
    }
  } else {
    if (x > 0.8) {
      state[SimVar::dens] = 33.0 / 62.0;
      state[SimVar::momx] = 0.0;
      state[SimVar::momy] = state[SimVar::dens] * (4. / sqrt(11.));
      state[SimVar::momz] = 0.0;
      state[SimVar::etot] = 0.3 / gamm + state[SimVar::dens] * (8.0 / 11.0);
    } else {
      state[SimVar::dens] = 77.0 / 558.0;
      state[SimVar::momx] = state[SimVar::dens] * (4.0 / sqrt(11.0));
      state[SimVar::momy] = state[SimVar::dens] * (4.0 / sqrt(11.0));
      state[SimVar::momz] = 0.0;
      state[SimVar::etot] = (9.0 / 310.0) / gamm + state[SimVar::dens] * (16.0 / 11.0);
    }
  }
}
