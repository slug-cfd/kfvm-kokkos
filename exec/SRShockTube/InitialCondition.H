// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::SRHydro::SimVar;
  
  (void) y;
  (void) z;
  
  const Real ax = Kokkos::fabs(x),gam = 5.0/3.0,gamm = gam - 1.0;
  const Real W = 1.0;
  const Real rho = ax<0.5 ? 10.0 : 1.0;
  const Real p = ax<0.5 ? 40.0/3.0 : 1.e-6;
  const Real h = 1.0 + (gam/gamm)*(p/rho);
  
  state[SimVar::dens] = rho*W;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = rho*h*W*W - p;
}
