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
  using Physics::MHD_GLM::SimVar;
  using Kokkos::sqrt;
  
  const Real gamm = eosParams.gamma - 1.0;
  const Real r0sq = 0.01;
  const Real rsq = SPACE_DIM == 2 ? x*x + y*y : x*x + y*y + z*z;
  const Real pG = rsq<=r0sq ? 1000.0 : 0.1;

  Real rho = 1.0,bx = 100.0/sqrt(4.0*M_PI);
  
  state[SimVar::dens] = rho;
  
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  
  state[SimVar::magx] = bx;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = 0.0;
  
  state[SimVar::psi] = 0.0;
  
  state[SimVar::etot] = pG/gamm + 0.5*bx*bx;
}
