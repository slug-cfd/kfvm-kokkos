// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;

  const Real gam = eosParams.gamma, gamm = gam - 1.0;
  const Real p = 1.0 / gam; // Start with unit soundspeed
  const Real bx = sqrt(2.0 * p / userParams.beta);

  state[ConsVar::dens] = 1.0;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm + 0.5 * bx * bx;
  state[ConsVar::magx] = bx;
  state[ConsVar::magy] = 0.0;
  state[ConsVar::magz] = 0.0;
}
