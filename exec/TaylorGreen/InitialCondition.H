// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,
		      EosParameters eosParams,
		      UserParameters userParams,
		      Real state[NUM_VARS])
{
  using Kokkos::sin;
  using Kokkos::cos;
  using Physics::Hydro::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  
  const Real L = 1.0,u0 = 1.0,rho = 1.0,p0 = 100.0;

  const Real u = u0*sin(x/L)*cos(y/L)*cos(z/L);
  const Real v = -u0*cos(x/L)*sin(y/L)*cos(z/L);
  const Real w = 0.0;
  const Real p = p0 + rho*(cos(2.0*x/L) + cos(2.0*y/L))*(cos(2.0*z/L) + 2.0)/16.0;
  
  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho*u;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = rho*w;
  state[SimVar::etot] = p/gamm + 0.5*rho*(u*u + v*v + w*w);
}
