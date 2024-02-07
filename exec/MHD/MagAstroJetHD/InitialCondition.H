// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  (void)x;
  (void)y;
  (void)z;

  const Real gam = eosParams.gamma, gamm = gam - 1.0;
  const Real dr = userParams.densityRatio, by = sqrt(2.0 / userParams.beta);

  state[ConsVar::dens] = gam / dr;

  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;

  state[ConsVar::magx] = 0.0;
  state[ConsVar::magy] = by;
  state[ConsVar::magz] = 0.0;

  state[ConsVar::etot] = 1.0 / gamm + 0.5 * by * by;
}
