// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::SRHydro::SimVar;

  (void)y;
  (void)z;

  const Real gR = eosParams.gamma / (eosParams.gamma - 1.0);
  const bool inside = (x < 1.0) && (y * y + z * z < 1.0);
  const Real v0 = 0.99, p = 0.01;

  Real rho, vx;
  if (inside) {
    rho = 0.01;
    vx = v0;
  } else {
    rho = 10.0;
    vx = 0.0;
  }

  const Real W = 1.0 / Kokkos::sqrt(1.0 - vx * vx), h = 1.0 + p * gR / rho;

  state[SimVar::dens] = rho * W;
  state[SimVar::momx] = rho * W * W * h * vx;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = rho * W * W * h - p;
}
