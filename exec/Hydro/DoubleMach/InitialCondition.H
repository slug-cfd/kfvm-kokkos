// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real r3 = sqrt(3.0); // slope of shock
  Real rho, u, v, w, p;

  if (y > r3 * (x - 1.0 / 6.0)) {
    rho = 8.0;
    u = 7.1447096;
    v = -4.125;
    w = 0.0;
    p = 116.5;
  } else {
    rho = 1.4;
    u = 0.0;
    v = 0.0;
    w = 0.0;
    p = 1.0;
  }

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0;
}
