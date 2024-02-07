// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// Dai and Woodward shock-bubble interaction

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real r0sq = 0.15 * 0.15;
  const Real rsq =
      (SPACE_DIM == 2)
          ? ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5))
          : ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));

  if (rsq < r0sq) {
    // Inside bubble
    state[ConsVar::dens] = 10.0;

    state[ConsVar::momx] = -10.0 * 11.2536;
    state[ConsVar::momy] = 0.0;
    state[ConsVar::momz] = 0.0;

    state[ConsVar::magx] = 0.0;
    state[ConsVar::magy] = 0.56418958;
    state[ConsVar::magz] = 0.56418958;

    state[ConsVar::etot] =
        1.0 / gamm + 0.5 * 10.0 * 11.2536 * 11.2536 + 0.56418958 * 0.56418958;
  } else if (x <= 0.6) {
    // Left post shock region
    state[ConsVar::dens] = 3.86859;

    state[ConsVar::momx] = 0.0;
    state[ConsVar::momy] = 0.0;
    state[ConsVar::momz] = 0.0;

    state[ConsVar::magx] = 0.0;
    state[ConsVar::magy] = 2.1826182;
    state[ConsVar::magz] = -2.1826182;

    state[ConsVar::etot] = 167.345 / gamm + 2.1826182 * 2.1826182;
  } else {
    // Right pre-shock region
    state[ConsVar::dens] = 1.0;

    state[ConsVar::momx] = -11.2536;
    state[ConsVar::momy] = 0.0;
    state[ConsVar::momz] = 0.0;

    state[ConsVar::magx] = 0.0;
    state[ConsVar::magy] = 0.56418958;
    state[ConsVar::magz] = 0.56418958;

    state[ConsVar::etot] = 1.0 / gamm + 0.5 * 11.2536 * 11.2536 + 0.56418958 * 0.56418958;
  }
}
