// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, UserParameters userParams,
                const Real state[NUM_VARS], Real force[NUM_VARS]) {
  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)state;
  (void)force;

  const Real gam = 5.0 / 3.0, gamm = gam - 1.0;
  const Real rho = state[0], u = state[1] / rho, v = state[2] / rho, w = state[3] / rho;
  const Real pa = 1.0, p = gamm * (state[4] - 0.5 * rho * (u * u + v * v + w * w));
  const Real Ta = userParams.densityRatio / gam, T = p / state[0];

  if (T > Ta) {
    force[4] = -8.776 * (p * p - pa * pa);
  }
}
