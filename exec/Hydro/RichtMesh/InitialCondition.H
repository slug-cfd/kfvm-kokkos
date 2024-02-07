// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)z;

  const Real g = eosParams.gamma, gm = g - 1.0, gp = g + 1.0;
  const Real m = userParams.mach;
  Real rho, u, v, w, p;

  if (x < -0.2) {
    rho = 1.0 / (1.0 - 2.0 * (1.0 - 1.0 / (m * m)) / gp);
    u = m * sqrt(g) * (1.0 - 1.0 / rho);
    v = 0.0;
    w = 0.0;
    p = 1.0 + 2 * g * (m * m - 1.0) / gp;
  } else if (x < y) {
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

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] = p / gm + rho * (u * u + v * v + w * w) / 2.0;
}
