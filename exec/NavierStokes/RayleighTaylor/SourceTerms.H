// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, UserParameters userParams,
                const Real state[NUM_VARS], Real force[NUM_VARS]) {
  using NavierStokes::SimVar;
  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)userParams;
  (void)state;
  (void)force;

  const Real fs = userParams.Froude * userParams.Froude;
  const Real g = SPACE_DIM == 2 ? 1.0 / fs : -1.0 / fs;

  force[SimVar::momy] = g * state[SimVar::dens];
  force[SimVar::etot] = g * state[SimVar::momy];
}