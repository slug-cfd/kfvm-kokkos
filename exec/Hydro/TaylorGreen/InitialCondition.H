// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::sin;
  using Physics::Hydro::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real gMasq = eosParams.gamma * userParams.mach * userParams.mach;

  const Real u = sin(x) * cos(y) * cos(z);
  const Real v = -cos(x) * sin(y) * cos(z);
  const Real w = 0.0;
  const Real rho =
      1.0 + gMasq * (cos(2.0 * x) + cos(2.0 * y)) * (cos(2.0 * z) + 2.0) / 16.0;
  const Real p =
      1.0 / gMasq + (cos(2.0 * x) + cos(2.0 * y)) * (cos(2.0 * z) + 2.0) / 16.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;
  state[SimVar::etot] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
}
