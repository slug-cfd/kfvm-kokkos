// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;

  Real rho, u, v, w, p;
  if (x < 0.0) {
    if (y < 0.0) {
      if (z < 0.0) {
        // bottom, south, west
        rho = 1.016216216;
        u = -0.401442839;
        v = -0.401442839;
        w = -0.401442839;
        p = 1.4;
      } else {
        // top, south, west
        rho = 0.8;
        u = 0.0;
        v = 0.0;
        w = -0.727606875;
        p = 1.0;
      }
    } else {
      if (z < 0.0) {
        // bottom, north, west
        rho = 0.8;
        u = 0.0;
        v = -0.727606875;
        w = 0.0;
        p = 1.0;
      } else {
        // top, north, west
        rho = 1.0;
        u = 0.727606875;
        v = 0.0;
        w = 0.0;
        p = 1.0;
      }
    }
  } else {
    if (y < 0.0) {
      if (z < 0.0) {
        // bottom, south, east
        rho = 0.8;
        u = -0.727606875;
        v = 0.0;
        w = 0.0;
        p = 1.0;
      } else {
        // top, south, east
        rho = 1.0;
        u = 0.0;
        v = 0.727606875;
        w = 0.0;
        p = 1.0;
      }
    } else {
      if (z < 0.0) {
        // bottom, north, east
        rho = 1.0;
        u = 0.0;
        v = 0.0;
        w = 0.727606875;
        p = 1.0;
      } else {
        // top, north, east
        rho = 0.53125;
        u = 0.0;
        v = 0.0;
        w = 0.0;
        p = 0.4;
      }
    }
  }

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0;
}
