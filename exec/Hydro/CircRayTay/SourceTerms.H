// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, EosParameters eosParams,
                UserParameters userParams, const Real state[NUM_VARS],
                Real force[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using Kokkos::acos;
  using Kokkos::sqrt;

  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;
  (void)state;
  (void)force;

  const Real rsq = x * x + y * y + z * z, r = sqrt(rsq), ri = 2.5 * sqrt(2.0);
  const Real theta = acos(z / r),
             phi = (y > 0 ? 1.0 : -1.0) * acos(x / sqrt(x * x + y * y));
  const Real g = rsq <= 1.0 ? -r : -1.0 / rsq;
  const Real gx = g * sin(theta) * cos(phi), gy = g * sin(theta) * sin(phi),
             gz = g * cos(theta);

  force[ConsVar::momx] = gx * state[ConsVar::dens];
  force[ConsVar::momy] = gy * state[ConsVar::dens];
  force[ConsVar::momz] = gz * state[ConsVar::dens];
  force[ConsVar::etot] =
      gx * state[ConsVar::momx] + gy * state[ConsVar::momy] + gz * state[ConsVar::momz];
}
