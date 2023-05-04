// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;
  using Kokkos::sqrt;
  using Kokkos::cos;

  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  const Real rho = y <= 0.0 ? 1.0 : 10.0;
  const Real p = y <= 3.0/5.0 - 0.1*rho*y;

  // velocity perturbations with "random sign"
  const Real rand = cos(1000.0*x)*cos(3333.0*y)*cos(505.0*z);
  const Real sv = rand > 0.0 ? 1.0 : -1.0;
  const Real v = 0.005*sv*(1.0 + cos(M_PI*y/0.1));
  
  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/gamm + 0.5*rho*v*v;
}
