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
  using Kokkos::pow;
  using Kokkos::cos;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoH = 10.0,rhoL = 1.0;
  const Real rho = y <= 0.0 ? 1.0 : 10.0;
  const Real p = y <= 0.0 ? 1.0 + 0.1*rhoH - y*rhoL : 1.0 + rhoH*(0.1 - y);
  const Real v = 0.01*cos(4.0*M_PI*x/0.1)*cos(6.0*M_PI*z/0.1)*pow(cos(M_PI*y/0.2),4);
  const Real bx = y <= 0.0 ? 0.0 : 1.e-6;
  const Real bz = y <= 0.0 ? 1.e-3 : 0.0;
  
  state[SimVar::dens] = rho;
  
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = rho*v;
  state[SimVar::momz] = 0.0;
  
  state[SimVar::magx] = bx;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = bz;
  
  state[SimVar::etot] = p/gamm + 0.5*rho*v*v + 0.5*(bx*bx + bz*bz);

  state[SimVar::psi] = 0.0;
}
