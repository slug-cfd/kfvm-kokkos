// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::MHD_GLM::SimVar;
  
  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  const Real rho = gam*gam,p = gam,eps = 0.2;

  const Real u = -(1.0 + eps*sin(2.0*M_PI*z))*sin(2.0*M_PI*y);
  const Real v = (1.0 + eps*sin(2.0*M_PI*z))*sin(2.0*M_PI*x);
  const Real w = eps*sin(2.0*M_PI*z);

  const Real bx = -sin(2.0*M_PI*y),by = sin(4.0*M_PI*x),bz = 0.0;
  
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
