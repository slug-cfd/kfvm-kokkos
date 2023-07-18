// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, const Real state[NUM_VARS],
                Real force[NUM_VARS]) {
  using Hydro::SimVar;
  using Kokkos::acos;
  using Kokkos::sqrt;

  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)state;
  (void)force;

  const Real r = sqrt(x * x + y * y), phi = atan2(y, x);
  const Real g = -1.0 / (r * r);
  const Real gx = g * cos(phi), gy = g * sin(phi);

  force[SimVar::momx] = gx * state[SimVar::dens];
  force[SimVar::momy] = gy * state[SimVar::dens];
  force[SimVar::etot] = gx * state[SimVar::momx] + gy * state[SimVar::momy];
}
