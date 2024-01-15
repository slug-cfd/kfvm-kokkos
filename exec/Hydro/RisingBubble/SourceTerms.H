// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, UserParameters userParams,
                const Real state[NUM_VARS], Real force[NUM_VARS]) {
  using Hydro::SimVar;
  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)state;
  (void)force;

  const Real g = -1.0;

  force[SimVar::momy] = g * state[SimVar::dens];
  force[SimVar::etot] = g * state[SimVar::momy];
}
