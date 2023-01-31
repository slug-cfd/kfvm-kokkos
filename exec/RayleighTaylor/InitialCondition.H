// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::EulerAdiabatic::SimVar;
  using Kokkos::Experimental::sqrt;
  using Kokkos::Experimental::cos;

  const Real gam = 1.4,gamm = 0.4;
  const Real rho = y <= 0.5 ? 2.0 : 1.0;
  const Real p = y <= 0.5 ? 2.0*y + 1.0 : y + 1.5;
  const Real c = y <= 0.5 ? sqrt(gam*(2.0*y + 1.0)/2.0) : sqrt(gam*(y + 1.5));
  const Real v = 0.025*c*cos(4.0*M_PI*x)*cos(4.0*M_PI*z);
  
  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/gamm + 0.5*rho*v*v;
}
