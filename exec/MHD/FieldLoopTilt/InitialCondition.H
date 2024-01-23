// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// ICs from Lee in http://dx.doi.org/10.1016/j.jcp.2013.02.049

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::fmin;
  using Kokkos::sqrt;
  using Physics::MHD_GLM::SimVar;

  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;

  const Real gamm = eosParams.gamma - 1.0;
  // Tilt angle cosine and sine
  const Real cw = 2.0 / sqrt(5.0), sw = 1.0 / sqrt(5.0);
  // Tilted x and radius with +- 1.0 offsets
  const Real xm = (x - 1.0) * cw + z * sw, rm = sqrt(xm * xm + y * y);
  const Real xr = x * cw + z * sw, rr = sqrt(xr * xr + y * y);
  const Real xp = (x + 1.0) * cw + z * sw, rp = sqrt(xp * xp + y * y);
  // Filter out for smallest radius and matching rotated x
  Real xx = xr, r = rr;
  if (rm < rr && rm < rp) { // rm smallest
    xx = xm;
    r = rm;
  } else if (rp < rr && rp < rm) { // rp smallest
    xx = xp;
    r = rp;
  } // else default is fine
  // Rotated magnetic field
  const Real bx = r < userParams.R ? -userParams.A0 * (y / r) * cw : 0.0;
  const Real by = r < userParams.R ? userParams.A0 * (xx / r) : 0.0;
  const Real bz = r < userParams.R ? -userParams.A0 * (y / r) * sw : 0.0;

  state[SimVar::dens] = 1.0;
  state[SimVar::momx] = 1.0;
  state[SimVar::momy] = 1.0;
  state[SimVar::momz] = 2.0;
  state[SimVar::etot] = 1.0 / gamm + 3.0 + 0.5 * (bx * bx + by * by + bz * bz);
  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = bz;
}
