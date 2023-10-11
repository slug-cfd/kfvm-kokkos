// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)y;
  (void)z;
  (void)userParams;

  // Parallel coordinate
  const Real xp = (2.0 * x + y) / sqrt(5.0);

  // "Left" state regions
  const bool left = (xp < 0.5) || (1.5 < xp && xp < 2.5) || (3.5 < xp);

  const Real gamm = eosParams.gamma - 1.0;

  state[SimVar::dens] = left ? 1.0 : 0.125;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = (left ? 1.0 : 0.1) / gamm;
}
