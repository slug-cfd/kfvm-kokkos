// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
Real Wendland(Real x,Real R)
{
  Real r = fabs(x)/R,rl = (1.0 - r)>0.0 ? (1.0 - r) : 0.0;
  return pow(rl,4)*(4.0*r + 1.0);
}

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  (void) z;
  
  Real xc = 0.0,yc = 0.0;
  Real R = 2.0;
  Real wx = Wendland(x - xc,R),wy = Wendland(y - yc,R);
  
  state[SimVar::dens] = 1.0 + wx*wy;
  state[SimVar::momx] = state[SimVar::dens];
  state[SimVar::momy] = 0.0; //state[SimVar::dens];
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = 1.0/0.4 + (state[SimVar::momx]*state[SimVar::momx])/(2.0*state[SimVar::dens]);
}
