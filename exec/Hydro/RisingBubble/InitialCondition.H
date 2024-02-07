// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoH = userParams.rhoH, rhoL = userParams.rhoL;
  const Real rsq = x * x + (y - 0.25) * (y - 0.25) + z * z;
  const Real rho = rsq < userParams.Rsq ? rhoL : rhoH;
  const Real p = 100.0 / eosParams.gamma - rhoH * y;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm;
}
