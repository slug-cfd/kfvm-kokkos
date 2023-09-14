// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)z;

  const Real g = eosParams.gamma, gm = g - 1.0, gp = g + 1.0;
  const Real m = userParams.mach;
  Real rho, u, v, w, p;

  if (x < 0.3) {
    rho = gp * m / (gm * m + 2.0);
    u = m * sqrt(g);
    v = 0.0;
    w = 0.0;
    p = (2.0 * g * m - gm) / gp;
  } else if (x < (y + 0.5)) {
    rho = 1.0;
    u = 0.0;
    v = 0.0;
    w = 0.0;
    p = 1.0;
  } else {
    rho = userParams.densRat;
    u = 0.0;
    v = 0.0;
    w = 0.0;
    p = 1.0;
  }

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;
  state[SimVar::etot] = p / gm + rho * (u * u + v * v + w * w) / 2.0;
}
