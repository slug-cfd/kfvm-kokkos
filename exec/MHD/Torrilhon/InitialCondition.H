// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// See Torrilhon in doi:10.1016/S0021-9991(03)00347-4 for ICs and
// exact solutions

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::sin;
  using ConsVar = typename SimVar<eqType>::ConsVar;

  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;

  const int ptype = userParams.probType;
  const Real gamm = eosParams.gamma - 1.0;

  Real alpha; // twist angle
  if (ptype == 0) {
    alpha = M_PI;
  } else if (ptype == 1) {
    alpha = 3.0;
  } else if (ptype == 2) {
    alpha = 2.4;
  } else if (ptype == 3) {
    alpha = 2.3;
  }

  Real rho, p, u, v, w, bx, by, bz;

  if (ptype <= 1) { // coplanar problem, ics (16) and (17) in paper
    u = 0.0;
    v = 0.0;
    w = 0.0;
    bx = 1.0;
    if (x < 0) {
      rho = 1.0;
      by = 1.0;
      bz = 0.0;
      p = 1.0;
    } else {
      rho = 0.2;
      by = cos(alpha);
      bz = sin(alpha);
      p = 0.2;
    }
  } else { // non-planar, ics (18) and (19) in paper
    u = 0.0;
    v = 0.0;
    bx = 1.1;
    if (x < 0) {
      rho = 1.7;
      w = 0.0;
      by = 1.0;
      bz = 0.0;
      p = 1.7;
    } else {
      rho = 0.2;
      w = -1.4968909;
      by = cos(alpha);
      bz = sin(alpha);
      p = 0.2;
    }
  }

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] =
      p / gamm + 0.5 * (rho * (u * u + v * v + w * w) + (bx * bx + by * by + bz * bz));
  state[ConsVar::magx] = bx;
  state[ConsVar::magy] = by;
  state[ConsVar::magz] = bz;
}
