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

  const Real a0 = 10.0, a = 10.0, b = 0.0;

  const Real r = sqrt(x * x + y * y + z * z), rxy = sqrt(x * x + y * y);
  const Real phi = atan2(y, x), theta = atan2(rxy, z);

  const Real rho = 1.0 + exp(-a * a * (r - b) * (r - b));
  const Real vr = a0 * r;
  const Real vx = vr * sin(theta) * cos(phi), vy = vr * sin(theta) * sin(phi);
  const Real vz = vr * cos(theta);
  const Real p = 1.0 / eosParams.gamma;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * vx;
  state[SimVar::momy] = rho * vy;
  state[SimVar::momz] = rho * vz;
  state[SimVar::etot] =
      p / (eosParams.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
}
