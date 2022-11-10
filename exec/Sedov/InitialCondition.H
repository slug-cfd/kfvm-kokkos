// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::EulerAdiabatic::SimVar;

  const Real R = 1.5/32.0,rsq = x*x + y*y + z*z;
  const Real ei = 0.4/(M_PI*R*R);

  // Background
  state[SimVar::dens] = 1.0;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = (1.e-5)/0.4;

  // Energy deposit
  if (rsq < R*R) {
    state[SimVar::etot] = ei;
  }
}
