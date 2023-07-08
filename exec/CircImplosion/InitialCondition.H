// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,
		      EosParameters eosParams,
		      UserParameters userParams,
		      Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;
  using Kokkos::pow;
  using Kokkos::cos;
  using Kokkos::atan2;

  const Real th = atan2(x,y),a0 = Real(1.0/120.0),m = 47.0;
  const Real Rsq1 = pow(Real(2.0/3.0) + a0*cos(m*th),2),Rsq2 = 0.64;
  const Real rsq = x*x + y*y;
  const Real rhoIn = 0.05,rhoMid = 1.0,rhoOut = 0.1;
  const Real eiIn = 0.15,eiMid = 0.15,eiOut = 15.0;
  
  state[SimVar::dens] = rsq <= Rsq1 ? rhoIn : (rsq <= Rsq2 ? rhoMid : rhoOut);
  state[SimVar::etot] = rsq <= Rsq1 ? eiIn : (rsq <= Rsq2 ? eiMid : eiOut);
  
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
}
