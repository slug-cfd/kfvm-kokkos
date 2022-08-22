// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
Real ivOmega(Real x,Real y,Real xc,Real yc,Real R,Real sig,Real beta)
{
  return beta*exp(-((x - xc)*(x - xc) + (y - yc)*(y - yc))/(2.0*sig*sig*R*R));
}

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  (void) z;
  
  Real xc = 0.0,yc = 0.0;
  Real R = 1.0,sig = 1.0,beta = 5.0*sqrt(2.0)*exp(0.5)/(4.0*M_PI);
  Real Om = ivOmega(x,y,xc,yc,R,sig,beta);
  
  state[SimVar::dens] = pow(1.0 + 0.4*Om*Om/2.0,1.0/0.4);
  state[SimVar::momx] = state[SimVar::dens];
  state[SimVar::momy] = state[SimVar::dens];
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = 1.0/0.4 + (state[SimVar::momx]*state[SimVar::momx] +
				   state[SimVar::momy]*state[SimVar::momy])/(2.0*state[SimVar::dens]);
}
