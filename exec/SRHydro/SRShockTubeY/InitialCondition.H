// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;

  (void)x;
  (void)z;

  const Real gR = eosParams.gamma / (eosParams.gamma - 1.0);
  const Real ay = Kokkos::fabs(y);
  const Real W = 1.0;
  const Real rho = ay < 0.5 ? 10.0 : 1.0;
  const Real p = ay < 0.5 ? 40.0 / 3.0 : 1.e-6;
  const Real h = 1.0 + p * gR / rho;

  state[ConsVar::dens] = rho * W;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = rho * W * W * h - p;
}
