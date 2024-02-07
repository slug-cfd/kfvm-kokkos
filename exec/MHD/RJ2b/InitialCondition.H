// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::fabs;
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0, trpi = 2.0 * sqrt(M_PI);

  const Real rhoL = 1.0, rhoR = 0.1;

  const Real uL = 0.0, uR = 0.0;
  const Real vL = 0.0, vR = 2.0;
  const Real wL = 0.0, wR = 1.0;

  const Real pL = 1.0, pR = 10.0;

  const Real bx = 3.0 / trpi;
  const Real byL = 6.0 / trpi, byR = 1.0 / trpi;
  const Real bzL = 0.0, bzR = 0.0;

  // Background
  state[ConsVar::magx] = bx;

  // Hi/Lo states
  if (fabs(x) <= 0.5) {
    state[ConsVar::dens] = rhoL;
    state[ConsVar::momx] = rhoL * uL;
    state[ConsVar::momy] = rhoL * vL;
    state[ConsVar::momz] = rhoL * wL;
    state[ConsVar::magy] = byL;
    state[ConsVar::magz] = bzL;
    state[ConsVar::etot] = pL / gamm + 0.5 * rhoL * (uL * uL + vL * vL + wL * wL) +
                          0.5 * (bx * bx + byL * byL + bzL * bzL);
  } else {
    state[ConsVar::dens] = rhoR;
    state[ConsVar::momx] = rhoR * uR;
    state[ConsVar::momy] = rhoR * vR;
    state[ConsVar::momz] = rhoR * wR;
    state[ConsVar::magy] = byR;
    state[ConsVar::magz] = bzR;
    state[ConsVar::etot] = pR / gamm + 0.5 * rhoR * (uR * uR + vR * vR + wR * wR) +
                          0.5 * (bx * bx + byR * byR + bzR * bzR);
  }
}
