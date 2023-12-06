// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// Dai and Woodward shock-bubble interaction

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::MHD_GLM::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real r0sq = 0.15 * 0.15;
  const Real rsq =
      (SPACE_DIM == 2)
          ? ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5))
          : ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));

  if (rsq < r0sq) {
    // Inside bubble
    state[SimVar::dens] = 10.0;

    state[SimVar::momx] = -10.0 * 11.2536;
    state[SimVar::momy] = 0.0;
    state[SimVar::momz] = 0.0;

    state[SimVar::magx] = 0.0;
    state[SimVar::magy] = 0.56418958;
    state[SimVar::magz] = 0.56418958;

    state[SimVar::etot] =
        1.0 / gamm + 0.5 * 10.0 * 11.2536 * 11.2536 + 0.56418958 * 0.56418958;
  } else if (x <= 0.6) {
    // Left post shock region
    state[SimVar::dens] = 3.86859;

    state[SimVar::momx] = 0.0;
    state[SimVar::momy] = 0.0;
    state[SimVar::momz] = 0.0;

    state[SimVar::magx] = 0.0;
    state[SimVar::magy] = 2.1826182;
    state[SimVar::magz] = -2.1826182;

    state[SimVar::etot] = 167.345 / gamm + 2.1826182 * 2.1826182;
  } else {
    // Right pre-shock region
    state[SimVar::dens] = 1.0;

    state[SimVar::momx] = -11.2536;
    state[SimVar::momy] = 0.0;
    state[SimVar::momz] = 0.0;

    state[SimVar::magx] = 0.0;
    state[SimVar::magy] = 0.56418958;
    state[SimVar::magz] = 0.56418958;

    state[SimVar::etot] = 1.0 / gamm + 0.5 * 11.2536 * 11.2536 + 0.56418958 * 0.56418958;
  }
}
