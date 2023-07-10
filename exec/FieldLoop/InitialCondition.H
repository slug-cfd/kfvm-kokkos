// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using Physics::MHD_GLM::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rho = 1.0, p = 1.0, A0 = 0.001;
  const Real R = 0.3, r = sqrt(x * x + y * y);

  const Real u = 2.0;
  const Real v = 1.0;
  const Real w = 0.0;

  const Real bx = (r < R ? -y * A0 / r : 0.0), by = (r < R ? x * A0 / r : 0.0), bz = 0.0;

  state[SimVar::dens] = rho;

  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;

  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = bz;

  state[SimVar::psi] = 0.0;

  state[SimVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0 +
                        (bx * bx + by * by + bz * bz) / 2.0;
}
