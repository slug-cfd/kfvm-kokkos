// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Kokkos::sin;
  using Kokkos::cos;
  using Kokkos::atan2;
  using Physics::Hydro::SimVar;
  
  (void) z;

  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  const Real rho = 1.0,V = 1.0;
  const Real theta = atan2(y,x),vx = -V*cos(theta),vy = -V*sin(theta);
  const Real P = 1.e-6;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho*vx;
  state[SimVar::momy] = rho*vy;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = P/gamm + 0.5*rho*(vx*vx + vy*vy);
  
}
