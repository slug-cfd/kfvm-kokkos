// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real r = sqrt(x * x + (y - 1.0) * (y - 1.0));
  const Real p = r < 0.2 ? 10.0 : 1.0;

  state[SimVar::dens] = 1.0;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / (eosParams.gamma - 1.0);
}
