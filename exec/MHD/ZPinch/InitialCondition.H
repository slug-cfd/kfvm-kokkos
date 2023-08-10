// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::sin;
  using Physics::MHD_GLM::SimVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rho = 1.0, V = 1.0;
  const Real theta = atan2(y, x), vx = -V * cos(theta), vy = -V * sin(theta);
  const Real P = 1.e-6, B = 1.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * vx;
  state[SimVar::momy] = rho * vy;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = P / gamm + 0.5 * rho * (vx * vx + vy * vy) + 0.5 * B * B;
  state[SimVar::magx] = 0.0;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = B;
  state[SimVar::psi] = 0.0;
}
