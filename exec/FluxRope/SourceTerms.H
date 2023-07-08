// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t,Real x,Real y,Real z,const Real state[NUM_VARS],Real force[NUM_VARS])
{
  using MHD_GLM::SimVar;
  
  (void) t; (void) x; (void) y; (void) z;
  (void) state; (void) force;

  force[SimVar::momy] = -state[SimVar::dens];
  force[SimVar::etot] = -state[SimVar::momy];
}
