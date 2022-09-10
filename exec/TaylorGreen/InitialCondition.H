// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{  
  const Real L = 1.0,u0 = 1.0,rho0 = 1.0,p0 = 1.0;//,p0 = 100.0;

  const Real u = u0*sin(x/L)*cos(y/L)*cos(z/L);
  const Real v = -u0*cos(x/L)*sin(y/L)*cos(z/L);
  const Real w = 0.0;
  const Real p = p0 + rho0*(cos(2.0*x/L) + cos(2.0*y/L))*(cos(2.0*z/L) + 2.0)/16.0;
  
  state[SimVar::dens] = rho0;
  state[SimVar::momx] = rho0*u;
  state[SimVar::momy] = rho0*v;
  state[SimVar::momz] = rho0*w;
  state[SimVar::etot] = p/0.4 + rho0*(u*u + v*v + w*w)/2.0;
}
