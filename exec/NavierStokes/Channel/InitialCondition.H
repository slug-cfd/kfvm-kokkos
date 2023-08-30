// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::NavierStokes::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real Masq = userParams.mach * userParams.mach;

  const Real rho = 1.0;
  const Real u = 1.0; // 4.0 * y * (1.0 - y);
  const Real v = 1.e-4 * y * (1.0 - y) * sin(2.0 * M_PI * x);
  const Real w = 0.0;
  const Real p = 1.0 / (gamm * eosParams.gamma * Masq);

  state[SimVar::dens] = rho;
  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = rho * w;
  state[SimVar::etot] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
}
