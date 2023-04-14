// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::SRHydro::SimVar;
  
  (void) y;
  (void) z;
  
  const Real ax = Kokkos::fabs(x),gam = 4.0/3.0,gamm = gam - 1.0;
  const Real vx = ax<0.5 ? 0.9 : 0.0,W = 1.0/Kokkos::sqrt(1.0 - vx*vx);
  const Real p = ax<0.5 ? 1.0 : 10.0;
  const Real h = 1.0 + p*gam/gamm;
  
  state[SimVar::dens] = W;
  state[SimVar::momx] = W*W*h*vx;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::tau] = W*W*h - p - W;
}
