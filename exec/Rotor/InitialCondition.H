// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::MHD_GLM::SimVar;
  using Kokkos::sqrt;
  
  const Real gam = 1.4,gamm = gam - 1.0;
  const Real r0 = 0.1,r1 = 0.115,r = sqrt(x*x + y*y + z*z);
  const Real fr = (r1 - r)/(r1 - r0);
  const Real pG = 1.0,u0 = 2.0;

  Real rho = 1.0,u = 0.0,v = 0.0,bx = 5.0/sqrt(4.0*M_PI);

  if (r <= r0) {
    rho = 10.0;
    u = -u0*y/r0;
    v = u0*x/r0;
  } else if (r <= r1) {
    rho = 1.0 + 9.0*fr;
    u = -fr*u0*y/r;
    v = fr*u0*x/r;
  }
  
  state[SimVar::dens] = rho;
  
  state[SimVar::momx] = rho*u;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = 0.0;
  
  state[SimVar::magx] = bx;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = 0.0;
  
  state[SimVar::psi] = 0.0;
  
  state[SimVar::etot] = pG/gamm + 0.5*rho*(u*u + v*v) + 0.5*bx*bx;
}
