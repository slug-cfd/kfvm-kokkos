// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, EosParameters eosParams,
                UserParameters userParams, const Real state[NUM_VARS],
                Real force[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;
  (void)state;
  (void)force;

  const Real gam = 5.0 / 3.0, gamm = gam - 1.0;
  const Real rho = state[ConsVar::dens], u = state[ConsVar::momx] / rho,
             v = state[ConsVar::momy] / rho, w = state[ConsVar::momz] / rho;
  const Real pa = 1.0,
             p = gamm * (state[ConsVar::etot] - 0.5 * rho * (u * u + v * v + w * w));
  const Real Ta = userParams.densityRatio / gam, T = p / rho;

  if (T > Ta) {
    force[ConsVar::etot] = -8.776 * (p * p - pa * pa);
  }
}
