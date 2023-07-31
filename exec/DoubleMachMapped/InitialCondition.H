// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  Real rho, u, v, w, p;

  if (x < 0.0) {
    rho = 8.0;
    u = 8.25;
    v = 0.0;
    w = 0.0;
    p = 116.5;
  } else {
    rho = 1.4;
    u = 0.0;
    v = 0.0;
    w = 0.0;
    p = 1.0;
  }

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;
  state[SimVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0;
}
