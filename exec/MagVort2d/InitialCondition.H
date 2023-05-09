// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::MHD_GLM::SimVar;
  using Kokkos::exp;
  
  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  const Real kappa = 5.0,mu = 5.0,tpi = 2.0*M_PI,rsq = x*x + y*y;
  const Real ex1 = exp(1.0 - rsq),ex2 = exp((1.0 - rsq)/2.0);
  
  const Real rho = 1.0;

  const Real u = 1.0 - y*kappa*ex2/tpi;
  const Real v = 1.0 + x*kappa*ex2/tpi;
  const Real w = 0.0;
  
  const Real bx = -y*mu*ex2/tpi;
  const Real by = x*mu*ex2/tpi;
  const Real bz = 0.0;

  const Real p = 1.0 + (kappa*kappa*(1.0 - rsq) - mu*mu)*ex1/(8.0*M_PI*M_PI);
  
  state[SimVar::dens] = rho;
  
  state[SimVar::momx] = rho*u;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = rho*w;
  
  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = bz;
  
  state[SimVar::psi] = 0.0;
  
  state[SimVar::etot] = p/gamm + rho*(u*u + v*v + w*w)/2.0 + (bx*bx + by*by + bz*bz)/2.0;
}
