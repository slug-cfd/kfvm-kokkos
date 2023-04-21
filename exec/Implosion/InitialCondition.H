// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;
  using Kokkos::fabs;
  
  if ((fabs(x) + fabs(y) + fabs(z)) <= 0.15) {
    state[SimVar::dens] = 0.125;
    state[SimVar::etot] = 0.14/0.4;
  } else {
    state[SimVar::dens] = 1.0;
    state[SimVar::etot] = 1.0/0.4;
  }
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
}
