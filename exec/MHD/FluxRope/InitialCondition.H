// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.718281828459045235
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::exp;
  using Kokkos::pow;
  using Kokkos::tanh;
  using Physics::MHD_GLM::SimVar;

  const Real gamm = eosParams.gamma - 1.0;

  // Layout anr field strengths
  const Real Bs = 0.0, Hb = 0.125, yc = 0.3;
  const Real Rsq = 0.125 * 0.125, rsq = x * x + (y - yc) * (y - yc), q = 2.5;

  // density and total pressure
  Real rho = 1.0;
  const Real pT = rho * (3.0 - y);
  rho = rsq <= Rsq ? 0.8 * rho : rho;

  // fluid velocities
  const Real v0 = 0.25;
  const Real preY = 0.5 * (1.0 + tanh(10.0 * (y - 0.75)));
  const Real preZ = 2.0 * sqrt(5.0 * M_E) * z * exp(-10.0 * z * z);
  const Real vx = v0 * preY * preZ, vy = 0.0, vz = 0.0;

  // Background magnetic field
  Real bx = Bs * exp((yc - y) / (2.0 * Hb)), by = 0.0, bz = 0.0;

  // Flux rope
  bx -= (rsq <= Rsq ? 2.0 * q * (y - yc) : 0.0);
  by += (rsq <= Rsq ? 2.0 * q * x : 0.0);
  bz += (rsq <= Rsq ? 1.0 : 0.0);

  // magnetic and gas pressure
  const Real pB = 0.5 * (bx * bx + by * by + bz * bz), pG = pT - pB;

  state[SimVar::dens] = rho;

  state[SimVar::momx] = rho * vx;
  state[SimVar::momy] = rho * vy;
  state[SimVar::momz] = rho * vz;

  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = bz;

  state[SimVar::etot] = pG / gamm + pB + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
}
