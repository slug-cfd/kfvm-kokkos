// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::MHD_GLM::SimVar;
  using Kokkos::Experimental::sqrt;
  
  const Real gam = 1.4,gamm = gam - 1.0;
  const Real r0 = 0.1,r = sqrt(x*x + y*y);
  const Real pG = r<=r0 ? 1000.0 : 0.1;

  Real rho = 1.0,bx = 100.0/sqrt(4.0*M_PI);
  
  state[SimVar::dens] = rho;
  
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  
  state[SimVar::magx] = bx;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = 0.0;
  
  state[SimVar::psi] = 0.0;
  
  state[SimVar::etot] = pG/gamm + 0.5*bx*bx;
}
