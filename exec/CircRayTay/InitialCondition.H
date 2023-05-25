// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
{
  using Physics::Hydro::SimVar;
  using Kokkos::sqrt;
  using Kokkos::acos;
  using Kokkos::cos;

  const Real gam = 5.0/3.0,gamm = gam - 1.0;
  const Real rhoH = 10.0,rhoL = 1.0;
  const Real R1 = 2.0,R2 = 4.0,P0 = 10.0;

  const Real rsq = x*x + y*y + z*z,r = sqrt(rsq),ri = 2.5*sqrt(2.0);
  const Real theta = acos(z/r),phi = (y>0 ? 1.0 : -1.0)*acos(x/sqrt(x*x + y*y));
  const Real g = rsq <= 1.0 ? -r : -1.0/rsq;
  const Real gx = g*sin(theta)*cos(phi),gy = g*sin(theta)*sin(phi),gz = g*cos(theta);
  
  Real rho,p;

  // Density with perturbation
  const Real A = 0.05*4.0*theta*(M_PI - theta)/(M_PI*M_PI);
  const Real rp = R1 + A*cos(10.0*phi)*sin(5.0*theta);
  if (r <= rp) {
    rho = rhoL;
  } else if (r <= R2) {
    rho = rhoH;
  } else {
    rho = rhoL;
  }

  // Pressure hydrostatic ignoring density perturbation
  if (r <= 1.0) {
    p = P0 - 0.5*rhoL*r;
  } else if (r <= R1) {
    p = P0 + rhoL*(1.0/r - 1.5);
  } else if (r <= R2) {
    p = P0 + rhoL*(1.0/R1 - 1.5) + rhoH*(1.0/r - 1.0/R1);
  } else {
    p = P0 + rhoL*(1.0/r + 1.0/R1 - 1.5 - 1.0/R2) + rhoH*(1.0/R2 - 1.0/R1);
  }
  
  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p/gamm;
}
