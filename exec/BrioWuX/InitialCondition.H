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
  using Kokkos::fabs;
  
  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoL = 1.0,rhoR = 0.125;
  const Real pL = 1.0,pR = 0.1;
  const Real byL = 1.0,byR = -1.0,bx = 0.75;
  
  // Background
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::magx] = bx;
  state[SimVar::magz] = 0.0;
  state[SimVar::psi] = 0.0;

  // Hi/Lo states
  if (fabs(x) <= 0.5) {
    state[SimVar::dens] = rhoL;
    state[SimVar::magy] = byL;
    state[SimVar::etot] = pL/gamm + 0.5*(bx*bx + byL*byL);
  } else {
    state[SimVar::dens] = rhoR;
    state[SimVar::magy] = byR;
    state[SimVar::etot] = pR/gamm + 0.5*(bx*bx + byR*byR);
  }
}
