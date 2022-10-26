// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  (void) z;
  Real p = 1.0 + sin(M_PI*(x - y))/2.0;
  
  state[SimVar::dens] = 1.0 + sin(M_PI*(x + y))/2.0;
  state[SimVar::momx] = state[SimVar::dens]*cos(M_PI*(x + 2.0*y));
  state[SimVar::momy] = state[SimVar::dens]*(1.0 - sin(M_PI*(2.0*x + y))/2.0);
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/0.4 +
    (state[SimVar::momx]*state[SimVar::momx] +
     state[SimVar::momy]*state[SimVar::momy])/(2.0*state[SimVar::dens]);
}
