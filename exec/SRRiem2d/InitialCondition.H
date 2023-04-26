// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::SRHydro::SimVar;
  
  (void) y;
  (void) z;
  
  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  Real rho,vx,vy,p;
  // Set primitives based on quadrant
  if (x > 0.0) {
    if (y > 0.0) {
      rho = 0.1;
      vx = 0.0;
      vy = 0.0;
      p = 0.01;
    } else {
      rho = 0.1;
      vx = 0.0;
      vy = 0.99;
      p = 1.0;
    }
  } else {
    if (y > 0.0) {
      rho = 0.1;
      vx = 0.99;
      vy = 0.0;
      p = 1.0;
    } else {
      rho = 0.5;
      vx = 0.0;
      vy = 0.0;
      p = 1.0;
    }
  }
  
  const Real W = 1.0/Kokkos::sqrt(1.0 - vx*vx - vy*vy),h = 1.0 + p*gam/(rho*gamm);
  
  state[SimVar::dens] = rho*W;
  state[SimVar::momx] = rho*W*W*h*vx;
  state[SimVar::momy] = rho*W*W*h*vy;
  state[SimVar::momz] = 0.0;
  state[SimVar::tau] = rho*W*(h*W - 1.0) - p;
}
