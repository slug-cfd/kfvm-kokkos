// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::exp;
  using Kokkos::sin;
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real rsq = (x - 0.5) * (x - 0.5) + z * z;
  const Real rho = 1.0 + userParams.rpf * exp(-100.0 * rsq);
  const Real vx = userParams.vx, vy = userParams.vy, vz = userParams.vz;
  const Real p = 1.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * vx;
  state[SimVar::momy] = rho * vy;
  state[SimVar::momz] = rho * vz;
  state[SimVar::etot] =
      p / (eosParams.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
}
