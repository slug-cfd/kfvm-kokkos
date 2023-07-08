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
  using Physics::LinAdv::SimVar;
  
  (void) x; (void) y; (void) z;

  state[SimVar::u] = cos(2.0*M_PI*x);
  
}
