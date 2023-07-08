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
  using Kokkos::cos;
  using Kokkos::sin;
  using Physics::Hydro::SimVar;
  
  (void) z;
  
  const Real gamm = eosParams.gamma - 1.0;

  const Real rho = 1.0 + sin(M_PI*(x + y))/2.0;
  const Real vx = cos(M_PI*(x + 2.0*y));
  const Real vy = (1.0 - sin(M_PI*(2.0*x + y))/2.0);
  const Real p = 1.0 + sin(M_PI*(x - y))/2.0;
  
  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho*vx;
  state[SimVar::momy] = rho*vy;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/gamm + 0.5*rho*(vx*vx + vy*vy);
}
