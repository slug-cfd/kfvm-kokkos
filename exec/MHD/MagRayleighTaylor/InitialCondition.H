// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoH = userParams.rhoH, rhoL = userParams.rhoL;
  const Real bc = sqrt(rhoH - rhoL), bx = userParams.bFac * bc;
#if (SPACE_DIM == 2)
  const Real rho = y < 0.5 ? rhoH : rhoL;
  const Real p = y < 0.5 ? rho * y + 1.0 : rho * y + 1.0 + 0.5 * (rhoH - rhoL);
  const Real c = sqrt(eosParams.gamma * p / rho);
  const Real v = x < 0.125 ? -0.025 * c * cos(8.0 * M_PI * x)
                           : -0.025 * c * cos(8.0 * M_PI * (0.25 - x));
#else
  const Real rho = y <= 0.0 ? rhoL : rhoH;
  const Real p = 100.0 / eosParams.gamma - rho * y;
  const Real v = -0.1 * exp(-25.0 * (x * x + z * z)) / pow(cosh(100.0 * y * y), 2.0);
#endif

  state[ConsVar::dens] = rho;

  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = 0.0;

  state[ConsVar::magx] = bx;
  state[ConsVar::magy] = 0.0;
  state[ConsVar::magz] = 0.0;

  state[ConsVar::etot] = p / gamm + 0.5 * rho * v * v + 0.5 * bx * bx;
}
