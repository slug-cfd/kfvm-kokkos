// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;
  
  (void) x;
  (void) y;
  Real az = Kokkos::fabs(z);
  state[SimVar::dens] = az<0.5 ? 1.0 : 0.125;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = (az<0.5 ? 1.0 : 0.1)/0.4;
}
