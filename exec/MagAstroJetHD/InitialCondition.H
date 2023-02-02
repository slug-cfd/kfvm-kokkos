// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef B0SQ
#define B0SQ 2000.0
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::MHD_GLM::SimVar;
  using Kokkos::Experimental::sqrt;
  
  (void) x; (void) y; (void) z;
  
  const Real gam = 1.4,gamm = gam - 1.0,b0 = sqrt(B0SQ);

  state[SimVar::dens] = gam/10.0;
  
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;

  state[SimVar::magx] = 0.0;
  state[SimVar::magy] = b0;
  state[SimVar::magz] = 0.0;
  
  state[SimVar::etot] = 1.0/gamm + b0*b0/2.0;
  
  state[SimVar::psi] = 0.0;
}
