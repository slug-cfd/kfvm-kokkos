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
  using Kokkos::sqrt;
  using Physics::MHD_GLM::SimVar;

  const Real gamm = eosParams.gamma - 1.0, lam = 1.0;
  const Real sina = 2.0 / 3.0, sinb = 2.0 / sqrt(5.0);
  const Real cosa = sqrt(5.0) / 3.0, cosb = 1.0 / sqrt(5.0);
  const Real x1 = x * cosa * cosb + y * cosa * sinb + z * sina;

  const Real rho = 1.0, p = 0.1;

  // Velocity and mag field in rotated coords
  const Real u1 = 0.0;
  const Real u2 = 0.1 * sin(2.0 * M_PI * x1 / lam);
  const Real u3 = 0.1 * cos(2.0 * M_PI * x1 / lam);
  const Real b1 = 1.0;
  const Real b2 = 0.1 * sin(2.0 * M_PI * x1 / lam);
  const Real b3 = 0.1 * cos(2.0 * M_PI * x1 / lam);

  // Rotate back
  const Real u = u1 * cosa * cosb - u2 * sinb - u3 * sina * cosb;
  const Real v = u1 * cosa * sinb + u2 * cosb - u3 * sina * sinb;
  const Real w = u1 * sina + u3 * cosa;
  const Real bx = b1 * cosa * cosb - b2 * sinb - b3 * sina * cosb;
  const Real by = b1 * cosa * sinb + b2 * cosb - b3 * sina * sinb;
  const Real bz = b1 * sina + b3 * cosa;

  state[SimVar::dens] = rho;

  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;

  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = bz;

  state[SimVar::psi] = 0.0;

  state[SimVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0 +
                        (bx * bx + by * by + bz * bz) / 2.0;
}
