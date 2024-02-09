// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::fabs;
  using ConsVar = typename SimVar<eqType>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoL = 1.0, rhoR = 0.125;
  const Real pL = 1.0, pR = 0.1;
  const Real bxL = 1.0, bxR = -1.0, by = 0.75;

  // Background
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::magy] = by;
  state[ConsVar::magz] = 0.0;

  // Hi/Lo states
  if (fabs(y) <= 0.5) {
    state[ConsVar::dens] = rhoL;
    state[ConsVar::magx] = bxL;
    state[ConsVar::etot] = pL / gamm + 0.5 * (bxL * bxL + by * by);
  } else {
    state[ConsVar::dens] = rhoR;
    state[ConsVar::magx] = bxR;
    state[ConsVar::etot] = pR / gamm + 0.5 * (bxR * bxR + by * by);
  }
}
