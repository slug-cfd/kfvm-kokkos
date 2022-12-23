// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#define ASTROJET_VEL 800.0

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::EulerAdiabatic::SimVar;
  
  (void) x; (void) y; (void) z;
  
  const Real gam = 1.4,gamm = gam - 1.0;

  const Real rsq = x*x + z*z,Rsq = 0.05*0.05;

  if (rsq > Rsq || y > 0.01) {
    state[SimVar::dens] = gam/10.0;
    state[SimVar::momx] = 0.0;
    state[SimVar::momy] = 0.0;
    state[SimVar::momz] = 0.0;
    state[SimVar::etot] = 1.0/gamm;
  } else {
    state[SimVar::dens] = gam;
    state[SimVar::momx] = 0.0;
    state[SimVar::momy] = gam*ASTROJET_VEL;
    state[SimVar::momz] = 0.0;
    state[SimVar::etot] = 1.0/gamm + gam*ASTROJET_VEL*ASTROJET_VEL/2.0;
  }
  
}
