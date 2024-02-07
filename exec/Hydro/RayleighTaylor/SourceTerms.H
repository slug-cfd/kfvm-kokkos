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

  const Real g = SPACE_DIM == 2 ? 1.0 : -1.0;

  force[ConsVar::momy] = g * state[ConsVar::dens];
  force[ConsVar::etot] = g * state[ConsVar::momy];
}
