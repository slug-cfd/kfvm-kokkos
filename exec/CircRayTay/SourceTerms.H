// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t,Real x,Real y,Real z,const Real state[NUM_VARS],Real force[NUM_VARS])
{
  using Hydro::SimVar;
  using Kokkos::sqrt;
  using Kokkos::acos;
  
  (void) t; (void) x; (void) y; (void) z;
  (void) state; (void) force;

  const Real rsq = x*x + y*y + z*z,r = sqrt(rsq),ri = 2.5*sqrt(2.0);
  const Real theta = acos(z/r),phi = (y>0 ? 1.0 : -1.0)*acos(x/sqrt(x*x + y*y));
  const Real g = rsq <= 1.0 ? -1.0 : -1.0/rsq;
  const Real gx = g*sin(theta)*cos(phi),gy = g*sin(theta)*sin(phi),gz = g*cos(theta);

  force[SimVar::momx] = gx*state[SimVar::dens];
  force[SimVar::momy] = gy*state[SimVar::dens];
  force[SimVar::momz] = gz*state[SimVar::dens];
  force[SimVar::etot] = gx*state[SimVar::momx] + gy*state[SimVar::momy] + gz*state[SimVar::momz];
}
