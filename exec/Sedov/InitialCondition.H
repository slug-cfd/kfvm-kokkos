// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#include "Definitions.H"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;

  const Real gam = 1.4,gamm = gam - 1.0;
  const Real R = SPACE_DIM == 2 ? 4.0/256.0 : 4.0/64.0; // 4 cell radius for typical resolutions
  const Real rsq = x*x + y*y + z*z;
  const Real p = rsq <= R*R ? gamm*gamm/(M_PI*R*R) : 1.e-5;

  // Background
  state[SimVar::dens] = 1.0;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/gamm;
}
