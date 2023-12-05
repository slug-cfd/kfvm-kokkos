// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoH = userParams.rhoH, rhoL = userParams.rhoL;
  const Real rsq = x * x + (y - 0.25) * (y - 0.25) + z * z;
  const Real rho = rsq < userParams.Rsq ? rhoL : rhoH;
  const Real p = 100.0 / eosParams.gamma - rhoH * y;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / gamm;
}
