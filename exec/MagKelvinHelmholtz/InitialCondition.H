// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Kokkos::tanh;
  using Kokkos::sin;
  using Kokkos::exp;
  using Physics::MHD_GLM::SimVar;
  
  (void) z;
  const Real gam = 1.4,gamm = gam - 1.0;
  Real rhoF = 1.0,p = 10.0,uf = 1.0,a = 0.05,sig = 0.2,A = 0.01,y1 = 0.5,y2 = 1.5,b = 0.1;
  
  state[SimVar::dens] = 1.0 + rhoF*(tanh((y-y1)/a) - tanh((y-y2)/a))/2.;
  
  state[SimVar::momx] = state[SimVar::dens]*uf*(tanh((y-y1)/a) - tanh((y-y2)/a) - 1);
  state[SimVar::momy] = state[SimVar::dens]*A*sin(2.0*M_PI*x)*(exp(-(y-y1)*(y-y1)/(sig*sig)) + exp(-(y-y2)*(y-y2)/(sig*sig)));
  state[SimVar::momz] = 0.0;
  
  state[SimVar::magx] = b*(tanh((y-y1)/a) - tanh((y-y2)/a) - 1);
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = 0.0;
  
  state[SimVar::etot] = p/gamm + state[SimVar::magx]*state[SimVar::magx]/2.0 +
    (state[SimVar::momx]*state[SimVar::momx] + state[SimVar::momy]*state[SimVar::momy])/(2.0*state[SimVar::dens]);

  state[SimVar::psi] = 0.0;
}
